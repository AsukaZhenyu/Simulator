"""Read-only replay of the preserved workload allocator capacity experiments.

This verifies tracked allocation capacity, not process/device physical capacity.
It never launches a native executable. Current source/runtime hashes are read,
so run this only outside an exclusive hardware timing campaign.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from profiling.workload_accuracy import contract as c
from profiling.workload_accuracy.run import build_binding
from profiling.workload_accuracy.tests import test_v2_evidence as route_audit

LABELS = ("baseline", "one_byte_below", "exact_peak", "one_byte_above")
SAMPLES, WARMUP = 3, 0


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def _json(text):
    def reject(value):
        raise ValueError("nonfinite JSON constant: " + value)
    return json.loads(text, object_pairs_hook=_object, parse_constant=reject)


def _load(path):
    return _json(Path(path).read_text(encoding="utf-8"))


def _inside(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("missing or escaping capacity evidence: " + name)
    return path


def _timestamp(value, label):
    if not isinstance(value, str):
        raise ValueError(label + " must be an ISO UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(label + " must be an ISO UTC timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(label + " must explicitly use UTC")
    return parsed


def _raw(root, label, binary):
    casepath = _inside(root, label + ".case.json")
    rawpath = _inside(root, label + ".raw.json")
    case = c.validate_case(_load(casepath))
    raw = _load(rawpath)
    c.same(raw.get("schema"), "workload_accuracy_raw/1", "capacity raw schema")
    c.same(raw.get("case"), case, "saved case versus raw case")
    expected = [str(binary), "--case", str(casepath), "--samples", str(SAMPLES),
                "--warmup", str(WARMUP)]
    c.same(raw.get("command"), expected, "expected capacity native command")
    c.same(raw.get("samples"), SAMPLES, "capacity sample count")
    c.same(raw.get("warmup"), WARMUP, "capacity warmup count")
    if raw.get("timed_out", False) is not False or "launch_error" in raw or "validation_error" in raw:
        raise ValueError("timeout, launch or validation failure cannot establish a capacity result")
    for field in ("stdout", "stderr"):
        if not isinstance(raw.get(field), str):
            raise ValueError("capacity raw " + field + " must be retained text")
    start = _timestamp(raw.get("started_utc"), "capacity start")
    end = _timestamp(raw.get("completed_utc"), "capacity completion")
    if end <= start:
        raise ValueError("capacity completion must follow start")
    return case, raw, start, end


def _allocations(payload):
    events = payload["memory"]["events"]
    births = {event["id"]: event for event in events if event["event"] == "allocate"}
    deaths = {event["id"]: event for event in events if event["event"] == "release"}
    if len(births) != sum(event["event"] == "allocate" for event in events):
        raise ValueError("allocation identity reused across expert lifetimes")
    if len(deaths) != len(births) or set(deaths) != set(births):
        raise ValueError("every tracked allocation requires exactly one release")
    return births, deaths


def _known_bytes(case, payload):
    """Recompute independent F32 resident payload sizes and graph-stage counts."""
    births, _ = _allocations(payload)
    h, width = case["hidden"], case["intermediate"]
    kd = case["kv_heads"] * case["head_dim"]
    base = 4 * (2*h + 2*h*h + 2*h*kd
                + (h*h if case["attention_gate"] == "sigmoid" else 0)
                + (2*case["head_dim"] if case["qk_norm"] == "rms" else 0))
    kv = 8 * kd * case["max_context"]
    for li, layer in enumerate(case["layers"]):
        expected = {"resident_weights": base, "kv_capacity": kv}
        if layer["kind"] == "dense":
            expected["resident_dense_ffn"] = 12*h*width
        for component, size in expected.items():
            matches = [e for e in births.values() if e["layer"] == li and e["component"] == component]
            c.same(len(matches), 1, "one persistent " + component + " payload")
            c.same(matches[0]["requested_bytes"], size, "independent F32 " + component + " bytes")
            metadata = [e for e in births.values() if e["layer"] == li
                        and e["component"] == component + ".metadata"]
            c.same(len(metadata), 1, "one persistent " + component + " metadata allocation")
        for event in births.values():
            if event["layer"] == li and re.fullmatch(r"resident_expert\.\d+", event["component"]):
                c.same(event["requested_bytes"], 12*h*width, "independent expert F32 payload bytes")
    count = 3 * len(case["layers"]) * len(case["rounds"])
    for component in ("graph_metadata", "graph_buffers"):
        c.same(sum(e["component"] == component for e in births.values()), count,
               "one " + component + " allocation per actual stage")
    c.same(sum(e["component"] == "cpu_workspace" for e in births.values()),
           count if case["backend"] == "CPU" else 0, "explicit CPU workspace allocation coverage")


def _expert_lifetimes(case, payload):
    """Reconstruct instances from current routes, then check actual reload IDs."""
    births, deaths = _allocations(payload)
    reload_observed = False
    instances_checked = 0
    for li, layer in enumerate(case["layers"]):
        if layer["kind"] != "moe":
            continue
        events = payload["residency"][li]["events"]
        snapshots = {e["round"]: e for e in events if e["event"] == "route_visible"}
        instances = []
        if layer["residency"] == "all":
            instances = [(e, -1, len(case["rounds"])) for e in range(layer["experts"])]
            if any(e["event"] in ("load", "unload") for e in events):
                raise ValueError("all-resident model setup must not masquerade as route-based loading")
        else:
            opened = {}
            for ri, rd in enumerate(case["rounds"]):
                active = {e for route in rd["routes"][li] for e in route}
                entering, leaving = active-set(opened), set(opened)-active
                for action, expected in (("load", entering), ("unload", leaving)):
                    observed = [e["expert"] for e in events if e["event"] == action and e["round"] == ri]
                    c.same(sorted(observed), sorted(expected), "route-derived exact expert " + action + " actions")
                for expert in sorted(leaving):
                    instances.append((expert, opened.pop(expert), ri))
                for expert in entering:
                    opened[expert] = ri
            instances.extend((expert, start, len(case["rounds"])) for expert, start in opened.items())
        expected_payload_ids = set()
        expected_metadata_ids = set()
        ids_by_expert = {}
        for expert, start, stop in instances:
            pair = {}
            for suffix in ("", ".metadata"):
                component = "resident_expert." + str(expert) + suffix
                matches = [e for e in births.values() if e["component"] == component
                           and e["layer"] == li and e["round"] == start]
                c.same(len(matches), 1, "one actual expert instance allocation: " + component)
                birth = matches[0]
                c.same(deaths[birth["id"]]["round"], stop, "route-derived actual expert release round")
                pair[suffix] = birth["id"]
            expected_payload_ids.add(pair[""])
            expected_metadata_ids.add(pair[".metadata"])
            ids_by_expert.setdefault(expert, []).append(pair[""])
            for ri in range(max(0, start), stop):
                c.same(snapshots[ri]["allocation_ids"][str(expert)], pair[""],
                       "resident snapshot references the actual live expert instance")
            instances_checked += 1
        actual_expert = [e for e in births.values() if e["layer"] == li
                         and e["component"].startswith("resident_expert.")]
        c.same(sorted(e["id"] for e in actual_expert), sorted(expected_payload_ids | expected_metadata_ids),
               "complete actual expert metadata and payload instance coverage")
        for identifiers in ids_by_expert.values():
            if len(set(identifiers)) != len(identifiers):
                raise ValueError("returning expert reused a released allocation identity")
            reload_observed = reload_observed or len(identifiers) > 1
    original_loads = [(layer["layer"], e["expert"]) for layer in payload["residency"]
                      for e in layer["events"] if e["event"] == "load"]
    c.same(len(original_loads) > len(set(original_loads)), reload_observed,
           "reported reload markers versus actual new expert allocations")
    return dict(expert_reload_observed=reload_observed, expert_instances_checked=instances_checked)


def _accepted(case, raw):
    if type(raw.get("returncode")) is not int or raw["returncode"] != 0:
        raise ValueError("baseline/exact/above native result must exit zero")
    payload = _json(raw["stdout"])
    c.validate_native(case, payload, samples=SAMPLES, warmup=WARMUP)
    c.same(payload["memory"]["budget_bytes"], case["tracked_budget_bytes"], "case versus native ledger budget")
    c.same(payload["memory"]["scope"], "tracked_backend_buffers_graph_metadata_explicit_cpu_work_only",
           "capacity accounting scope")
    route = route_audit.audit_expert_allocation_order(case, payload)
    _known_bytes(case, payload)
    lifetimes = _expert_lifetimes(case, payload)
    return payload, {**route, **lifetimes}


def verify_capacity(directory):
    """Return a receipt after read-only replay; do not write into evidence."""
    root = Path(directory).resolve()
    binding_path = _inside(root, "binding.json")
    report_path = _inside(root, "capacity-report.json")
    binding, report = _load(binding_path), _load(report_path)
    if not isinstance(binding, dict) or set(binding) != {"native", "python_sources"}:
        raise ValueError("capacity native/Python source binding is missing")
    binary = Path(binding["native"]["binary"]).resolve()
    c.same(binding["native"].get("schema"), "workload_accuracy_build/1", "native build binding schema")
    c.same(build_binding(binary), binding, "current capacity build/source/runtime binding")
    for suffix in (".case.json", ".raw.json"):
        c.same(sorted(p.name for p in root.glob("*" + suffix)),
               sorted(label + suffix for label in LABELS), "exact preserved capacity case coverage")
    records = {}
    previous_end = None
    files = {binding_path.name: c.sha(binding_path), report_path.name: c.sha(report_path)}
    for label in LABELS:
        case, raw, start, end = _raw(root, label, binary)
        if previous_end is not None and start < previous_end:
            raise ValueError("capacity runs do not follow baseline/below/exact/above order")
        previous_end = end
        records[label] = (case, raw)
        for suffix in (".case.json", ".raw.json"):
            files[label + suffix] = c.sha(root / (label + suffix))
    baseline, baseline_raw = records["baseline"]
    native, lifecycle = _accepted(baseline, baseline_raw)
    peak = c.integer(native["memory"]["peak_tracked_reserved_bytes"], "tracked baseline peak", 2)
    if baseline["tracked_budget_bytes"] < peak:
        raise ValueError("baseline peak exceeds its original declared budget")
    rows = []
    positive_checks = {"baseline": lifecycle}
    expected_budgets = {"one_byte_below": peak-1, "exact_peak": peak, "one_byte_above": peak+1}
    for label, expected_budget in expected_budgets.items():
        case, raw = records[label]
        expected_case = copy.deepcopy(baseline)
        expected_case["tracked_budget_bytes"] = expected_budget
        c.same(case, expected_case, "same capacity workload with exact declared " + label + " budget")
        accepted = label != "one_byte_below"
        if accepted:
            trial, checked = _accepted(case, raw)
            c.same(trial["memory"]["peak_tracked_reserved_bytes"], peak, "same actual tracked allocation peak")
            c.same(trial["memory"]["events"], native["memory"]["events"],
                   "same actual allocator components, sizes and lifetimes under changed budget")
            positive_checks[label] = checked
        else:
            if type(raw.get("returncode")) is not int or raw["returncode"] != 1 or raw["stdout"].strip():
                raise ValueError("one-byte-below must be native exit 1 with no success payload")
            crossing = next((event for event in native["memory"]["events"]
                             if event["event"] == "allocate" and event["live_reserved_bytes"] > expected_budget), None)
            if crossing is None:
                raise ValueError("below-budget rejection has no corresponding baseline crossing allocation")
            message = "ggml-workload-accuracy-probe: tracked_budget_exceeded before allocation: " + crossing["component"]
            if raw["stderr"].splitlines().count(message) != 1:
                raise ValueError("one-byte-below did not reject before the expected crossing allocation")
        rows.append(dict(case=label, budget_bytes=expected_budget, expected_tracked_acceptance=accepted,
                         actual_tracked_acceptance=raw["returncode"] == 0, physical_capacity_verified=False))
    rebuilt = dict(schema="workload_accuracy_capacity/1", case_sha256=c.digest(baseline),
                   tracked_peak_bytes=peak, rows=rows, passed=True, allocation_release_checked=True,
                   expert_reload_observed=lifecycle["expert_reload_observed"],
                   physical_capacity_verified=False, process_peak_bytes=None,
                   cuda_private_workspace_bytes=None, scope="explicit_allocator_payload_budget_only")
    c.same(report, rebuilt, "exact reconstructed capacity report")
    c.same(build_binding(binary), binding, "current capacity binding unchanged during offline replay")
    return dict(schema="workload_capacity_verification_receipt/1", verified=True,
                evidence_directory=str(root), evidence_file_sha256=files,
                native_binary_sha256=binding["native"]["binary_sha256"],
                build_binding_sha256=c.digest(binding), case_sha256=c.digest(baseline),
                backend=baseline["backend"], tracked_peak_bytes=peak,
                current_build_source_runtime_binding_verified=True,
                exact_commands_cases_exit_status_and_sampling_verified=True,
                exact_below_equal_above_budgets_verified=True,
                expected_preallocation_rejection_verified=True,
                numeric_shape_known_bytes_lifetimes_and_reloads_verified=True,
                metadata_and_payload_causality_verified=True,
                capacity_report_exactly_reconstructed=True,
                positive_case_audits=positive_checks,
                verifier_source_sha256={str(Path(__file__).resolve()): c.sha(__file__),
                                       str(Path(route_audit.__file__).resolve()): c.sha(route_audit.__file__)},
                frozen_evidence_modified=False, native_execution_performed=False,
                physical_capacity_verified=False, process_peak_bytes=None,
                cuda_private_workspace_bytes=None, scope="explicit_allocator_payload_budget_only")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, help="optional new receipt path; existing files are never overwritten")
    args = parser.parse_args()
    try:
        receipt = verify_capacity(args.directory)
        encoded = json.dumps(receipt, indent=2, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(encoded)
        print(encoded, end="")
        return 0
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        print("capacity verification: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
