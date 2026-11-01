"""Independent stdlib artifact audit of the hardware-system campaign.

No simulator, compiler, strategy, cost-adapter or campaign modules are imported.
This validates internal evidence and controls, not real hardware accuracy.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time


class AuditError(ValueError):
    pass


def need(condition, message):
    if not condition:
        raise AuditError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def within(directory, relative):
    path = (directory / relative).resolve()
    need(path.is_relative_to(directory.resolve()), "artifact path escapes its directory")
    return path


def ceil_scaled(duration, alpha):
    need(type(duration) is int and duration > 0, "nonpositive reference task duration")
    factor = Fraction(str(alpha))
    need(factor > 0, "nonpositive proxy multiplier")
    return (duration * factor.numerator + factor.denominator - 1) // factor.denominator


def alias_classes(refs):
    labels = {}
    return tuple((tensor, location, labels.setdefault(identity, len(labels)))
                 for (tensor, location), identity in sorted(refs.items()))


def union(spans):
    result = []
    for left, right in sorted(spans):
        if left == right:
            continue
        if result and left <= result[-1][1]:
            result[-1] = (result[-1][0], max(right, result[-1][1]))
        else:
            result.append((left, right))
    return result


def length(spans):
    return sum(right - left for left, right in spans)


def intersection(left, right):
    return sum(max(0, min(b, d) - max(a, c)) for a, b in left for c, d in right)


def audit_trace(events, operation_rows, tensor_rows, summary, hardware):
    """Replay readiness, physical aliases, resources and memory from events.

    The retained final catalogues give graph ports and exact declared prices.
    This independent replay does not trust final copy states or memory peaks.
    """
    operations = {row["id"]: row for row in operation_rows}
    tensors = {row["id"]: row for row in tensor_rows}
    need(len(operations) == len(operation_rows) and len(tensors) == len(tensor_rows),
         "duplicate graph identifiers")
    shared = hardware["memory_mode"] == "shared"
    pools = {"dram": "shared" if shared else "dram", "vram": "shared" if shared else "vram"}
    capacity = ({"shared": hardware["shared_capacity_bytes"]} if shared else
                {"dram": hardware["dram_capacity_bytes"], "vram": hardware["vram_capacity_bytes"]})
    reserve = {pool: 0 for pool in capacity}
    if shared:
        reserve["shared"] = hardware.get("shared_runtime_reserved_bytes", 0)
        need(hardware["dram_capacity_bytes"] == hardware["vram_capacity_bytes"] == capacity["shared"],
             "shared compatibility capacities do not refer to one pool")
    sizes = {tid: dict(row["allocation_bytes"]) for tid, row in tensors.items()}
    status = {(tid, loc): ("READY" if loc in row["initial_locations"] else "ABSENT")
              for tid, row in tensors.items() for loc in ("dram", "vram", "ssd")}
    refs, serial = {}, 0
    for tid in sorted(tensors):
        identity = serial
        for loc in ("dram", "vram"):
            if status[tid, loc] == "READY":
                refs[tid, loc] = (pools[loc], identity if shared else serial)
                serial += 1
    op_status = {key: "NOT_STARTED" for key in operations}
    running, now = [], 0
    peak = dict(reserve)
    payload = Counter()
    kinds, spans = Counter(), defaultdict(list)
    compute_counts = Counter()
    zero_release = 0

    def finish_at(moment):
        nonlocal running, now
        need(type(moment) is int and moment >= now, "event clock is not monotone")
        remaining = []
        for task in running:
            if task["end"] > moment:
                remaining.append(task)
            else:
                for destination in task["destinations"]:
                    status[destination] = "READY"
                if task["op"] is not None:
                    op_status[task["op"]] = "DONE"
        running, now = remaining, moment

    def check_memory():
        current = Counter(reserve)
        by_allocation = {}
        for (tid, loc), identity in refs.items():
            need(status[tid, loc] != "ABSENT", "dangling physical allocation")
            pool = pools[loc]
            value = (tid, sizes[tid][loc])
            need(identity not in by_allocation or by_allocation[identity] == value,
                 "allocation aliases distinct tensors or sizes")
            by_allocation[identity] = value
        for (pool, _), (_, size) in by_allocation.items():
            current[pool] += size
        for task in running:
            current[task["pool"]] += task["workspace"]
        for pool in capacity:
            need(0 <= current[pool] <= capacity[pool], f"{pool} capacity exceeded")
            peak[pool] = max(peak[pool], current[pool])

    check_memory()
    for index, event in enumerate(events):
        need(event["action_index"] == index, "event index is not contiguous")
        finish_at(event["t_ns"])
        kind, target = event["kind"], event.get("target_id")
        kinds[kind] += 1
        need(event.get("bytes_transferred", 0) >= 0 and event.get("bytes_released", 0) >= 0,
             "negative transfer or release amount")
        if kind == "ADVANCE":
            need(event.get("resource") is None and event.get("bytes_transferred", 0) == 0,
                 "ADVANCE performs physical work")
        elif kind == "SHARE":
            src, dst = event["src"], event["dst"]
            need(shared and {src, dst} == {"dram", "vram"}, "SHARE crosses physical pools")
            need(event["start_ns"] == event["end_ns"] == now and event.get("resource") is None
                 and event.get("bytes_transferred", 0) == 0 and event.get("workspace_bytes", 0) == 0,
                 "SHARE has physical time, payload or workspace")
            need(status[target, src] == "READY" and status[target, dst] == "ABSENT",
                 "SHARE source not ready or destination already present")
            status[target, dst] = "READY"
            refs[target, dst] = refs[target, src]
        elif kind == "EVICT":
            loc = event["location"]
            need(loc in pools and status[target, loc] == "READY", "eviction targets unavailable view")
            identity = refs[target, loc]
            need(not any(identity in task["reads"] for task in running), "eviction during physical reader")
            del refs[target, loc]
            released = 0 if identity in refs.values() else sizes[target][loc]
            need(event["bytes_released"] == released, "eviction release double-counts aliases")
            zero_release += released == 0
            status[target, loc] = "ABSENT"
        elif kind == "COMPUTE":
            need(target in operations and op_status[target] == "NOT_STARTED", "invalid or repeated computation")
            op, device = operations[target], event["device"]
            need(device in ("cpu", "gpu"), "unknown executor or unimplemented third device")
            loc = "dram" if device == "cpu" else "vram"
            need(all(status[tid, loc] == "READY" for tid in set(op["inputs"])),
                 "computation consumes an unready input")
            price = next((p for p in op["prices"] if p["device"] == device), None)
            need(price is not None, "execution device lacks declared price")
            need(event["start_ns"] == now and event["end_ns"] - now == price["duration_ns"] > 0,
                 "compute time does not match exact task price")
            need(event["workspace_bytes"] == price["allocated_workspace_bytes"], "workspace differs from task price")
            resource = device + "_compute"
            need(event["resource"] == resource and not any(task["resource"] == resource for task in running),
                 "compute resource overlaps another task")
            destinations = [(tid, loc) for tid in op["outputs"]]
            need(all(all(status[tid, domain] == "ABSENT" for domain in pools) for tid, _ in destinations),
                 "compute overwrites a live output")
            reads = {refs[tid, loc] for tid in set(op["inputs"])}
            for tid, domain in destinations:
                refs[tid, domain] = (pools[domain], serial)
                serial += 1
                status[tid, domain] = "RESERVED_OUTPUT"
            running.append(dict(end=event["end_ns"], destinations=destinations, reads=reads,
                resource=resource, workspace=event["workspace_bytes"], pool=pools[loc], op=target))
            op_status[target] = "RUNNING"
            compute_counts[target] += 1
            spans[device].append((now, event["end_ns"]))
        elif kind in ("COPY", "READ_SSD", "WRITE_SSD"):
            src, dst = event["src"], event["dst"]
            need((src, dst) in (("dram", "vram"), ("vram", "dram"), ("ssd", "dram"), ("dram", "ssd")),
                 "unknown physical copy route")
            need(status[target, src] == "READY" and status[target, dst] == "ABSENT", "COPY readiness invalid")
            price = next((p for p in tensors[target]["transfers"] if p["src"] == src and p["dst"] == dst), None)
            need(price is not None and event["start_ns"] == now
                 and event["end_ns"] - now == price["duration_ns"] > 0, "COPY has no positive exact price")
            need(event["bytes_transferred"] == tensors[target]["storage_bytes"], "COPY payload differs from tensor")
            resource = "ssd_io" if "ssd" in (src, dst) else "pcie_copy"
            need(event["resource"] == resource and not any(task["resource"] == resource for task in running),
                 "COPY resource overlap")
            reads = set() if src == "ssd" else {refs[target, src]}
            if dst != "ssd":
                refs[target, dst] = (pools[dst], serial)
                serial += 1
            status[target, dst] = "RESERVED_COPY"
            running.append(dict(end=event["end_ns"], destinations=[(target, dst)], reads=reads,
                resource=resource, workspace=0, pool=pools.get(dst, "shared" if shared else "dram"), op=None))
            payload[src + "->" + dst] += event["bytes_transferred"]
            spans["copy"].append((now, event["end_ns"]))
        else:
            raise AuditError(f"unsupported event kind {kind}")
        check_memory()

    if summary.get("success"):
        need(not running and all(state == "DONE" for state in op_status.values()), "successful run has unfinished tasks")
        need(set(compute_counts) == set(operations) and all(count == 1 for count in compute_counts.values()),
             "successful run omitted or repeated operations")
        need(summary["completed_makespan_ns"] == now, "final event clock differs from latency")
    for tid, tensor in tensors.items():
        need(dict(tensor["copies"]) == {loc: status[tid, loc] for loc in ("dram", "vram", "ssd")},
             "final tensor readiness differs from replay")
        if summary.get("success") and tensor.get("goal_location"):
            need(status[tid, tensor["goal_location"]] == "READY", "final tensor goal not delivered")
    if shared:
        emitted = {(tid, loc): identity for tid, tensor in tensors.items()
                   for loc, identity in tensor.get("allocation_refs", []) if identity is not None}
        need(alias_classes(emitted) == alias_classes(refs), "final physical alias relation differs from replay")
        need(summary["peak_shared_bytes"] == summary["peak_dram_bytes"] == summary["peak_vram_bytes"] == peak["shared"],
             "shared peak double-counts logical views")
        need(summary.get("physical_memory_peaks_bytes") == {"shared": peak["shared"]}, "shared physical summary has duplicate pools")
    else:
        need(summary["peak_dram_bytes"] == peak["dram"] and summary["peak_vram_bytes"] == peak["vram"],
             "independent memory peak differs from replay")
    for key, route in (("h2d_bytes", "dram->vram"), ("d2h_bytes", "vram->dram"),
                       ("ssd_read_bytes", "ssd->dram"), ("ssd_write_bytes", "dram->ssd")):
        need(summary.get(key, 0) == payload[route], "transfer byte conservation failed")
    need(summary["event_counts"] == dict(kinds), "event-kind summary differs from replay")
    if summary.get("success"):
        cpu, gpu, copy = (union(spans[name]) for name in ("cpu", "gpu", "copy"))
        compute = union(cpu + gpu)
        expected = dict(cpu_event_span_ns=length(cpu), gpu_event_span_ns=length(gpu),
            copy_event_span_ns=length(copy), compute_event_span_ns=length(compute),
            cpu_gpu_event_overlap_ns=intersection(cpu, gpu),
            copy_compute_event_overlap_ns=intersection(compute, copy),
            any_engine_event_span_ns=length(union(compute + copy)),
            all_engine_event_uncovered_ns=now - length(union(compute + copy)))
        need(all(summary[key] == value for key, value in expected.items()), "event-span metrics differ from independent unions")
    return dict(events=len(events), compute_events=sum(compute_counts.values()), share_events=kinds["SHARE"],
                copy_events=kinds["COPY"] + kinds["READ_SSD"] + kinds["WRITE_SSD"],
                zero_release_events=zero_release, replayed_physical_peaks_bytes=peak)


def audit_proxy_costs(reference, proxy, alpha):
    """Independently verify exact prices and unchanged physical declarations."""
    baseline_files = {path.name: path for path in (reference / "inputs").glob("*.costs.json")}
    proxy_files = {path.name: path for path in (proxy / "inputs").glob("*.costs.json")}
    need(baseline_files.keys() == proxy_files.keys(), "proxy missing cost-table shape")
    counts = Counter()
    for name, path in baseline_files.items():
        left, right = read(path), read(proxy_files[name])
        need(left.get("transfer_entries") == right.get("transfer_entries"), "proxy changes COPY costs")
        need(len(left["entries"]) == len(right["entries"]), "proxy changes supported task shapes")
        for base, changed in zip(left["entries"], right["entries"]):
            expected = dict(base)
            if base["device"] == "gpu":
                expected["duration_ns"] = ceil_scaled(base["duration_ns"], alpha)
                counts["proxy_accelerator_prices_checked"] += 1
            else:
                counts["proxy_cpu_prices_checked"] += 1
            need(expected == changed, "proxy changes shape, precision, scratch, signatures or nonaccelerator prices")
        if alpha == 1:
            need(path.read_bytes() == proxy_files[name].read_bytes(), "alpha=1 costs are not byte-identical")
            counts["alpha_one_cost_files_byte_identical"] += 1
    left, right = read(reference / "declarations.json"), read(proxy / "declarations.json")
    for key in ("model", "hardware", "workload", "scheduling"):
        need(left[key] == right[key], f"proxy changes paired {key}")
    metadata = read(proxy / "inputs/analytical-metadata.json")
    claim = metadata["npu_proxy"]
    need(claim["alpha"] == alpha and claim["execution_slot"] == "gpu"
         and claim["measured"] is False and claim["simultaneous_gpu_and_npu"] is False,
         "proxy metadata claims native device support or wrong sensitivity")
    need(metadata["calibrated"] is False and metadata["hardware_measurements_used"] is False,
         "synthetic proxy mislabeled measured")
    return counts


def audit_round_boundaries(events, operation_rows, global_document, actual):
    owners = {}
    grouped = {entry["id"]: [] for entry in global_document["rounds"]}
    for entry in global_document["rounds"]:
        for invocation in entry["invocations"]:
            need(invocation["id"] not in owners, "duplicate invocation round ownership")
            owners[invocation["id"]] = entry["id"]
    operations = {operation["id"]: operation for operation in operation_rows}
    for event in events:
        if event["kind"] == "COMPUTE":
            invocation = operations[event["target_id"]]["invocation_id"]
            need(invocation in owners, "compute has no round owner")
            grouped[owners[invocation]].append(event)
    need({row["round_id"] for row in actual} == {round_id for round_id, rows in grouped.items() if rows},
         "round-boundary coverage incomplete")
    for row in actual:
        events_in_round = grouped[row["round_id"]]
        expected = dict(compute_count=len(events_in_round),
            first_compute_start_ns=min(event["start_ns"] for event in events_in_round),
            last_compute_completion_ns=max(event["end_ns"] for event in events_in_round),
            cpu_compute_duration_sum_ns=sum(event["end_ns"] - event["start_ns"]
                for event in events_in_round if event["device"] == "cpu"),
            accelerator_compute_duration_sum_ns=sum(event["end_ns"] - event["start_ns"]
                for event in events_in_round if event["device"] == "gpu"))
        need(all(row[key] == value for key, value in expected.items()), "round compute boundary disagrees with events")
    return len(actual)


def identity(row):
    return row["system_id"], row["variant_id"], row["workload_id"], row["candidate_id"]


def audit_campaign(campaign, root, *, expected_attempts=480, check_current_sources=True,
                   legacy_manifest=None):
    campaign, root = Path(campaign).resolve(), Path(root).resolve()
    started, issues, counts = time.perf_counter(), [], Counter()
    receipt = dict(format_version="hardware-system-independent-audit-0.1", passed=False,
                   campaign=str(campaign), issues=issues, counts=counts)
    try:
        completion, protocol = read(campaign / "completion.json"), read(campaign / "protocol.json")
        records = read(campaign / "records.json")
        need(completion["complete"] is True and not protocol.get("plan_only"), "campaign incomplete or plan only")
        need(protocol["calibrated"] is False and protocol["hardware_measurement"] is False
             and protocol["native_npu_backend"] is False, "protocol claims unsupported measurements or native NPU")
        need(len(records) == completion["completed_attempts"] == completion["declared_attempts"]
             == protocol["declared_attempts"] == expected_attempts, "wrong attempted case count")
        need(canonical(records) == completion["records_sha256"], "records do not match completion hash")
        need(canonical(protocol) == completion["protocol_sha256"], "protocol does not match completion hash")
        before, after = read(campaign / "source-manifest-before.json"), read(campaign / "source-manifest-after.json")
        need(before == after and canonical(before) == completion["source_manifest_sha256"]
             == protocol["execution_sources_sha256"], "executed sources changed during campaign")
        if check_current_sources:
            for relative, digest in before.items():
                need(sha(within(root, relative)) == digest, f"current execution source changed: {relative}")
                counts["execution_source_files_hashed"] += 1
        accuracy = protocol["frozen_accuracy_evidence"]
        need(accuracy["available"] is True and len(accuracy["files"]) == 9, "missing frozen accuracy evidence")
        for file in accuracy["files"]:
            need(sha(within(root, file["path"])) == file["sha256"], "frozen accuracy evidence changed")
            counts["frozen_accuracy_files_hashed"] += 1
        runtime_manifest = (Path(legacy_manifest).resolve() if legacy_manifest is not None else
            root / "build/shared-storage-20261023/legacy-regression-v3/source-manifest-after.json")
        legacy = read(runtime_manifest)
        need(len(legacy) > 100, "legacy runtime baseline is incomplete")
        for relative, digest in legacy.items():
            need(sha(within(root, relative)) == digest, f"runtime differs from passed 352 regression baseline: {relative}")
            counts["legacy_execution_baseline_files_unchanged"] += 1
            if relative.startswith(("src/llm_simulator/", "modeling/llm_infer_model/", "mapping/tensor_mapping/")):
                counts["legacy_runtime_baseline_files_unchanged"] += 1
            if relative.startswith("modeling/llm_infer_model/performance/"):
                counts["performance_model_files_unchanged"] += 1
        systems = {row["id"]: row for row in protocol["system_catalogue"]}
        workloads = {row["spec"]["id"]: row for row in protocol["workloads"]}
        need(len(systems) == 9 and len(workloads) == 4, "wrong system/workload catalogue width")
        native_expected = {(system, "gpu", workload, f"e{e}-s{s}")
                           for system in systems for workload in workloads for e in range(4) for s in range(2)}
        proxy_ids = {job["system_id"] for job in protocol["jobs"] if job["accelerator"] == "npu_proxy"}
        need(proxy_ids == {"core_ultra258v", "snapdragon8elite_phone"}, "unexpected proxy systems")
        proxy_expected = {(system, variant, workload, f"e{e}-s{s}") for system in proxy_ids
                          for variant in ("npu-proxy-a0p5", "npu-proxy-a1", "npu-proxy-a2")
                          for workload in workloads for e in range(4) for s in range(2)}
        expected = native_expected | proxy_expected
        actual = [identity(row) for row in records]
        need(len(actual) == len(set(actual)) and set(actual) == expected, "missing, duplicated or undeclared case")
        jobs = {(job["system_id"], "gpu" if job["accelerator"] == "gpu" else
                "npu-proxy-a" + format(job["npu_time_multiplier"], "g").replace(".", "p"),
                job["workload_id"], job["candidate_id"]) for job in protocol["jobs"]}
        need(jobs == expected and len(protocol["jobs"]) == len(expected), "predeclared job matrix differs from independent matrix")
        indexed = dict(zip(actual, records))
        streamed = [json.loads(line) for line in (campaign / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
        need(len(streamed) == len(records) and {identity(row): row for row in streamed} == indexed,
             "streamed rows disagree with final records")
        for row in records:
            need(row["calibrated"] is False and row["hardware_measurement"] is False
                 and row["native_npu_backend"] is False and row["gpu_npu_concurrency"] is False
                 and row["numerical_llm_execution"] is False and row["npu_operator_support_validated"] is False,
                 "result claims unsupported measurements or capabilities")
            folder = within(campaign, row["artifact_directory"])
            need(read(folder / "summary.json") == row, "case summary differs from records")
            manifest = read(folder / "artifacts.json")
            files = {path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file()
                     and path.name != "artifacts.json"}
            need(files == set(manifest), "case artifact manifest omits or adds files")
            for relative, digest in manifest.items():
                need(sha(within(folder, relative)) == digest, "case artifact hash mismatch")
                counts["case_artifact_files_hashed"] += 1
            declarations = read(folder / "declarations.json")
            need(canonical(declarations) == row["declared_inputs_sha256"], "declaration hash mismatch")
            need(canonical(declarations["scheduling"]) == row["strategy_sha256"], "strategy hash mismatch")
            need(canonical({key: declarations[key] for key in ("model", "hardware", "workload")})
                 == row["workload_hardware_sha256"], "workload/hardware hash mismatch")
            workload = workloads[row["workload_id"]]
            need(declarations["model"] == workload["model"] and declarations["workload"] == workload["workload"],
                 "strategy input differs from fixed workload")
            expected_hardware = dict(memory_mode="separate", shared_capacity_bytes=None,
                shared_runtime_reserved_bytes=0, shared_alignment_bytes=1)
            expected_hardware.update(systems[row["system_id"]]["hardware"])
            need(declarations["hardware"] == expected_hardware, "case hardware differs from predeclared system")
            declared_strategy = protocol["strategies_by_system"][row["system_id"]][row["model_kind"]][row["candidate_id"]]
            need(declarations["scheduling"] == declared_strategy, "case strategy differs from predeclared strategy")
            hw = declarations["hardware"]
            budget = (hw["vram_capacity_bytes"] if hw["memory_mode"] == "separate" else
                      hw["shared_capacity_bytes"] - hw["shared_runtime_reserved_bytes"])
            need(declarations["scheduling"]["cache"]["budget_bytes"] == budget
                 == protocol["cache_budget_by_system"][row["system_id"]], "cache budget differs from available physical pool")
            if not row.get("success"):
                need(row["completed_makespan_ns"] is None, "failed case presented as completed latency")
                counts["explicit_failed_cases"] += 1
            if (folder / "events.json").exists():
                events = read(folder / "events.json")
                need(canonical(events) == row["event_sha256"], "event digest mismatch")
                need(canonical(read(folder / "compiled-ir.json")) == row["compiled_ir_sha256"], "IR digest mismatch")
                metadata = read(folder / "inputs/analytical-metadata.json")
                need(metadata["hardware"] == declarations["hardware"] and metadata["model"] == declarations["model"]
                     and metadata["calibrated"] is False and metadata["hardware_measurements_used"] is False,
                     "metadata inputs/provenance mismatch")
                need(read(folder / "inputs/environment.json")["events"] == [], "undeclared dynamic resource trajectory")
                operations = read(folder / "operation-catalogue.json")
                replay = audit_trace(events, operations,
                                     read(folder / "tensor-catalogue.json"), row, declarations["hardware"])
                if row["memory_mode"] == "shared":
                    need(row["h2d_bytes"] == row["d2h_bytes"] == 0, "alias-only campaign has phantom host/device copy")
                if row.get("success"):
                    boundaries = read(folder / "round-compute-boundaries.json")
                    need(boundaries == row["round_compute_boundaries"], "row/file round boundaries disagree")
                    counts["round_compute_boundaries_checked"] += audit_round_boundaries(
                        events, operations, read(folder / "inputs/model.global.json"), boundaries)
                for name in ("events", "compute_events", "share_events", "copy_events", "zero_release_events"):
                    counts[name + "_replayed"] += replay[name]
                counts["physical_memory_dependency_replays"] += 1
                counts["shared_memory_replays" if row["memory_mode"] == "shared" else "separate_memory_replays"] += 1
            counts["case_rows_checked"] += 1
        for row in records:
            if row["accelerator"] != "npu_proxy":
                continue
            reference = indexed[row["system_id"], "gpu", row["workload_id"], row["candidate_id"]]
            alpha = row["npu_time_multiplier"]
            counts.update(audit_proxy_costs(within(campaign, reference["artifact_directory"]),
                                           within(campaign, row["artifact_directory"]), alpha))
            counts["proxy_price_case_pairs_checked"] += 1
            if alpha == 1 or row["candidate_id"].startswith("e1-"):
                a, b = within(campaign, row["artifact_directory"]), within(campaign, reference["artifact_directory"])
                need(read(a / "events.json") == read(b / "events.json"), "proxy control has different actual events")
                for field in ("success", "completed_makespan_ns", "h2d_bytes", "d2h_bytes", "peak_dram_bytes", "peak_vram_bytes", "peak_shared_bytes"):
                    need(row.get(field) == reference.get(field), "proxy control has different outcome")
                counts["unique_event_control_pairs_checked"] += 1
                if alpha == 1:
                    counts["alpha_one_event_pairs_checked"] += 1
                if row["candidate_id"].startswith("e1-"):
                    counts["cpu_only_event_pairs_checked"] += 1
        need(counts["alpha_one_event_pairs_checked"] == 64 and counts["cpu_only_event_pairs_checked"] == 48
             and counts["unique_event_control_pairs_checked"] == 96, "control coverage incomplete")
        controls = read(campaign / "controls.json")
        need(len(controls) == 96 and all(control["equivalent"] for control in controls), "campaign controls incomplete")
        need(sum(bool(row["success"]) for row in records) == completion["successful_runs"]
             and sum(not row["success"] for row in records) == completion["failed_runs"], "completion outcomes disagree")
        receipt.update(passed=True, native_cases=len(native_expected), proxy_cases=len(proxy_expected),
            attempts_checked=len(records), completed_campaign_sha256=sha(campaign / "completion.json"),
            records_sha256=sha(campaign / "records.json"), protocol_sha256=sha(campaign / "protocol.json"),
            source_manifest_sha256=canonical(before), legacy_runtime_manifest=str(runtime_manifest),
            legacy_runtime_manifest_sha256=sha(runtime_manifest))
    except (AuditError, KeyError, OSError, ValueError, TypeError) as error:
        issues.append(f"{type(error).__name__}: {error}")
    receipt.update(counts=dict(counts), independent_audit_script_sha256=sha(Path(__file__)),
        wall_seconds=time.perf_counter() - started,
        limitations=["Artifact integrity and internal semantic consistency, not real hardware performance accuracy.",
            "No simulator, compiler, campaign, cost-adapter or hardware catalogue code imported.",
            "Graph port readiness and physical memory are replayed; no proof of real NPU operator support or production LLM correctness."])
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--legacy-manifest", type=Path,
        help="explicit archived legacy 352 source manifest; default is the existing local baseline, never skipped")
    args = parser.parse_args(argv)
    need(not args.output.resolve().is_relative_to(args.campaign.resolve()), "audit output must be outside immutable campaign")
    receipt = audit_campaign(args.campaign, args.root, legacy_manifest=args.legacy_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                           encoding="utf-8", newline="\n")
    print(json.dumps(receipt, ensure_ascii=False))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
