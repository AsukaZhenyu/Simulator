"""Synthetic offline capacity-verifier tests; never launch native/hardware work.

The toy numerical payload intentionally certifies the evidence contract only.
It is fabricated, carries mock binding metadata and is never hardware evidence.
"""
from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from profiling.accuracy_suite import verify_capacity as verifier
from profiling.workload_accuracy import contract as c
from profiling.workload_accuracy.examples import case as example_case


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def synthetic_native(case):
    events, residency = [], [[] for _ in case["layers"]]
    live, current, peak, sequence, serial = {}, 0, 0, 0, 0
    opened = [{} for _ in case["layers"]]
    persistent = []

    def allocate(component, size, ri, li):
        nonlocal current, peak, sequence, serial
        ident = "allocation" + str(serial)
        serial += 1
        current += size
        peak = max(peak, current)
        event = dict(event="allocate", sequence=sequence, id=ident, component=component,
                     requested_bytes=size, reserved_bytes=size, live_reserved_bytes=current,
                     round=ri, layer=li)
        sequence += 1
        events.append(event)
        live[ident] = event
        return ident

    def release(ident, ri, li):
        nonlocal current, sequence
        birth = live.pop(ident)
        current -= birth["reserved_bytes"]
        events.append(dict(birth, event="release", sequence=sequence,
                           live_reserved_bytes=current, round=ri, layer=li))
        sequence += 1

    def mark(li, ri, action, **kwargs):
        nonlocal sequence
        residency[li].append(dict(event=action, sequence=sequence, round=ri, **kwargs))
        sequence += 1

    def graph(ri, li):
        meta = allocate("graph_metadata", 128, ri, li)
        arena = allocate("graph_buffers", 64, ri, li)
        work = allocate("cpu_workspace", 0, ri, li)
        release(arena, ri, li)
        release(work, ri, li)
        release(meta, ri, li)

    h, width = case["hidden"], case["intermediate"]
    kd = case["kv_heads"] * case["head_dim"]
    base_bytes = 4 * (2*h + 2*h*h + 2*h*kd)
    for li in range(len(case["layers"])):
        for component, size in (("resident_weights.metadata", 16), ("resident_weights", base_bytes),
                                ("kv_capacity.metadata", 16), ("kv_capacity", 8*kd*case["max_context"])):
            persistent.append(allocate(component, size, -1, li))
    rows, rounds = [], []
    history = 0
    for ri, rd in enumerate(case["rounds"]):
        n = len(rd["hidden_input"])
        x = [struct.unpack("f", struct.pack("f", value))[0] for row in rd["hidden_input"] for value in row]
        for li, layer in enumerate(case["layers"]):
            graph(ri, li)
            mark(li, ri, "projection_complete")
            graph(ri, li)
            mark(li, ri, "attention_complete")
            mark(li, ri, "route_known", routes=rd["routes"][li],
                 selected_scores=rd["route_scores"][li], learned_router_executed=False)
            active = {expert for route in rd["routes"][li] for expert in route}
            leaving, entering = set(opened[li])-active, active-set(opened[li])
            for expert in sorted(leaving):
                mark(li, ri, "unload", expert=expert)
                metadata, payload = opened[li].pop(expert)
                release(payload, ri, li)
                release(metadata, ri, li)
            for expert in sorted(entering):
                metadata = allocate(f"resident_expert.{expert}.metadata", 16, ri, li)
                payload = allocate(f"resident_expert.{expert}", 12*h*width, ri, li)
                opened[li][expert] = (metadata, payload)
                mark(li, ri, "load", expert=expert)
            mark(li, ri, "route_visible", active_experts=sorted(active), resident_experts=sorted(active),
                 allocation_ids={str(e): opened[li][e][1] for e in sorted(active)},
                 information_scope="current_layer_round_only")
            mark(li, ri, "expert_ffn_begin")
            graph(ri, li)
            mark(li, ri, "expert_ffn_complete")
            sizes = dict(q=h*n, k_new=kd*n, v_new=kd*n, persistent_k=kd*(history+n),
                         persistent_v=kd*(history+n), attention_output=h*n, hidden_output=h*n)
            rows.append(dict(round=ri, layer=li, kind="moe", tokens=n,
                             history_before=history, history_after=history+n,
                             kv_used_bytes=8*kd*(history+n), kv_capacity_bytes=8*kd*case["max_context"],
                             checks=[dict(port=name, elements=size, max_abs_error=0., relative_l2=0., passed=True)
                                     for name, size in sizes.items()],
                             stages=[dict(kind=kind, compute_ns=10) for kind in ("projection", "attention", "moe_ffn_trace")],
                             hidden_input=x[:], hidden_output=x[:], reference_output=x[:]))
        rounds.append(dict(round=ri, hidden_output=x[:]))
        history += n
    for li, group in enumerate(opened):
        for metadata, payload in group.values():
            release(payload, len(case["rounds"]), -1)
            release(metadata, len(case["rounds"]), -1)
    for ident in reversed(persistent):
        release(ident, len(case["rounds"]), -1)
    stages = [dict(round=row["round"], layer=row["layer"], kind=stage["kind"], compute_ns=10)
              for row in rows for stage in row["stages"]]
    compute = sum(stage["compute_ns"] for stage in stages)
    timing = dict(full_workload_wall_ns=compute+1000, stage_compute_sum_ns=compute,
                  host_setup_materialization_gap_ns=1000,
                  scope="inference_graph_setup_alloc_materialization_weight_lifecycle_release",
                  scalar_reference_in_timed_region=False, port_serialization_in_timed_region=False,
                  memory_event_logging_in_timed_region=False, source_weight_generation_in_timed_region=False)
    memory = dict(scope="tracked_backend_buffers_graph_metadata_explicit_cpu_work_only",
                  budget_bytes=case["tracked_budget_bytes"], peak_tracked_reserved_bytes=peak,
                  final_tracked_reserved_bytes=0, events=events, tracked_budget_pass=True,
                  physical_capacity_verified=False, process_peak_bytes=None,
                  cuda_private_workspace_bytes=None, cuda_private_workspace_bound_known=False)
    return dict(schema="ggml_workload_accuracy/2", case_id=case["id"], backend="CPU", cuda_device=0,
                hardware_compute_performed=True, numeric_pass=True, dtype="f32", samples=3, warmup=0,
                each_sample_resets_workload_state=True,
                execution="actual_input_trace_multilayer_persistent_kv",
                causal_route_boundary="trace_known_after_attention_before_route_based_residency_and_ffn",
                learned_router_executed=False,
                precision=dict(matmul="GGML_PREC_F32", NVIDIA_TF32_OVERRIDE=None, GGML_CUDA_DISABLE_GRAPHS=None),
                rows=rows, rounds=rounds, memory=memory,
                residency=[dict(layer=li, final_history=history, events=value) for li, value in enumerate(residency)],
                sample_timings=[dict(sample=s, timing=copy.deepcopy(timing), stages=copy.deepcopy(stages),
                                     output_checks_after_timing=[dict(passed=True) for _ in case["rounds"]],
                                     peak_tracked_reserved_bytes=peak) for s in range(3)])


def fixture(root):
    binary = root / "synthetic-not-executable.bin"
    binary.write_bytes(b"SYNTHETIC ONLY; never executed")
    source = root / "synthetic-source.py"
    source.write_text("# SYNTHETIC binding fixture only\n", encoding="utf-8")
    source_map = {str(source.resolve()): c.sha(source)}
    binding = dict(native=dict(schema="workload_accuracy_build/1", binary=str(binary.resolve()),
                               binary_sha256=c.sha(binary), sources=source_map,
                               ggml_headers=source_map, runtime=source_map, link_libraries=source_map),
                   python_sources=source_map)
    write(root / "binding.json", binding)
    baseline = example_case("moe")
    native = synthetic_native(baseline)
    peak = native["memory"]["peak_tracked_reserved_bytes"]
    crossing = next(e for e in native["memory"]["events"] if e["event"] == "allocate" and e["live_reserved_bytes"] > peak-1)
    start = datetime(2026, 10, 18, tzinfo=timezone.utc)
    budgets = dict(baseline=baseline["tracked_budget_bytes"], one_byte_below=peak-1,
                   exact_peak=peak, one_byte_above=peak+1)
    for index, (label, budget) in enumerate(budgets.items()):
        case = copy.deepcopy(baseline)
        case["tracked_budget_bytes"] = budget
        casepath = (root / (label + ".case.json")).resolve()
        write(casepath, case)
        accepted = label != "one_byte_below"
        trial = copy.deepcopy(native)
        trial["memory"]["budget_bytes"] = budget
        raw = dict(schema="workload_accuracy_raw/1", case=case,
                   command=[str(binary.resolve()), "--case", str(casepath), "--samples", "3", "--warmup", "0"],
                   started_utc=(start + timedelta(seconds=index)).isoformat(),
                   completed_utc=(start + timedelta(seconds=index, milliseconds=500)).isoformat(),
                   returncode=0 if accepted else 1, stdout=json.dumps(trial) if accepted else "",
                   stderr="" if accepted else "ggml-workload-accuracy-probe: tracked_budget_exceeded before allocation: " + crossing["component"] + "\n",
                   samples=3, warmup=0)
        write(root / (label + ".raw.json"), raw)
    report = dict(schema="workload_accuracy_capacity/1", case_sha256=c.digest(baseline), tracked_peak_bytes=peak,
                  rows=[dict(case=label, budget_bytes=budgets[label], expected_tracked_acceptance=label != "one_byte_below",
                             actual_tracked_acceptance=label != "one_byte_below", physical_capacity_verified=False)
                        for label in ("one_byte_below", "exact_peak", "one_byte_above")],
                  passed=True, allocation_release_checked=True, expert_reload_observed=True,
                  physical_capacity_verified=False, process_peak_bytes=None, cuda_private_workspace_bytes=None,
                  scope="explicit_allocator_payload_budget_only")
    write(root / "capacity-report.json", report)
    return binding


class CapacityVerifierTests(unittest.TestCase):
    def setUp(self):
        parent = Path(__file__).resolve().parent
        self.temporary = tempfile.TemporaryDirectory(prefix="synthetic-capacity-", dir=parent)
        self.root = Path(self.temporary.name).resolve()
        self.assertTrue(self.root.is_relative_to(parent))
        self.binding = fixture(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def verify(self):
        with patch.object(verifier, "build_binding", return_value=copy.deepcopy(self.binding)):
            return verifier.verify_capacity(self.root)

    def mutate_raw(self, label, change):
        path = self.root / (label + ".raw.json")
        raw = json.loads(path.read_text(encoding="utf-8"))
        change(raw)
        write(path, raw)

    def mutate_native(self, change):
        def mutate(raw):
            value = json.loads(raw["stdout"])
            change(value)
            raw["stdout"] = json.dumps(value)
        self.mutate_raw("baseline", mutate)

    def test_complete_synthetic_replay_preserves_all_input_bytes(self):
        before = {p.name: c.sha(p) for p in self.root.iterdir() if p.is_file()}
        receipt = self.verify()
        self.assertTrue(receipt["verified"])
        self.assertTrue(receipt["metadata_and_payload_causality_verified"])
        self.assertTrue(receipt["positive_case_audits"]["baseline"]["expert_reload_observed"])
        self.assertFalse(receipt["physical_capacity_verified"])
        self.assertFalse(receipt["native_execution_performed"])
        self.assertEqual(before, {p.name: c.sha(p) for p in self.root.iterdir() if p.is_file()})

    def test_expected_command_and_saved_case_binding(self):
        self.mutate_raw("exact_peak", lambda raw: raw["command"].__setitem__(-1, "1"))
        with self.assertRaisesRegex(ValueError, "expected capacity native command"):
            self.verify()

    def test_same_workload_only_budget_may_change(self):
        path = self.root / "exact_peak.case.json"
        value = json.loads(path.read_text()); value["seed"] += 1
        write(path, value)
        self.mutate_raw("exact_peak", lambda raw: raw.__setitem__("case", value))
        with self.assertRaisesRegex(ValueError, "same capacity workload"):
            self.verify()

    def test_success_requires_zero_exit_and_no_timeout(self):
        self.mutate_raw("exact_peak", lambda raw: raw.__setitem__("returncode", 1))
        with self.assertRaisesRegex(ValueError, "must exit zero"):
            self.verify()

    def test_below_budget_does_not_accept_timeout_or_launch_failure(self):
        self.mutate_raw("one_byte_below", lambda raw: raw.__setitem__("timed_out", True))
        with self.assertRaisesRegex(ValueError, "timeout, launch"):
            self.verify()

    def test_below_budget_requires_expected_preallocation_failure(self):
        self.mutate_raw("one_byte_below", lambda raw: raw.__setitem__("stderr", "out of device memory\n"))
        with self.assertRaisesRegex(ValueError, "expected crossing allocation"):
            self.verify()

    def test_report_must_exactly_reconstruct_budgets_and_unknowns(self):
        path = self.root / "capacity-report.json"
        report = json.loads(path.read_text()); report["rows"][1]["budget_bytes"] += 1
        write(path, report)
        with self.assertRaisesRegex(ValueError, "exact reconstructed capacity report"):
            self.verify()

    def test_known_kv_payload_bytes_are_independently_recomputed(self):
        def mutate(value):
            birth = next(e for e in value["memory"]["events"] if e["event"] == "allocate" and e["component"] == "kv_capacity")
            ident = birth["id"]
            for event in value["memory"]["events"]:
                if event["id"] == ident:
                    event["requested_bytes"] -= 4
        self.mutate_native(mutate)
        with self.assertRaisesRegex(ValueError, "independent F32 kv_capacity bytes"):
            self.verify()

    def test_metadata_and_payload_early_allocation_corruptions(self):
        original = json.loads((self.root / "baseline.raw.json").read_text())
        for metadata in (True, False):
            write(self.root / "baseline.raw.json", original)
            def mutate(value):
                known = next(e["sequence"] for e in value["residency"][0]["events"] if e["event"] == "route_known" and e["round"] == 0)
                birth = next(e for e in value["memory"]["events"] if e["event"] == "allocate" and e["layer"] == 0 and e["round"] == 0
                             and e["component"].startswith("resident_expert.") and e["component"].endswith(".metadata") == metadata)
                birth["sequence"] = known-1
            self.mutate_native(mutate)
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                self.verify()

    def test_reload_markers_must_match_actual_route_transitions(self):
        def mutate(value):
            event = next(e for e in value["residency"][0]["events"] if e["event"] == "load" and e["round"] == 2)
            event["expert"] = 3
        self.mutate_native(mutate)
        with self.assertRaisesRegex(ValueError, "route-derived exact expert load"):
            self.verify()

    def test_current_build_binding_drift_is_rejected(self):
        changed = copy.deepcopy(self.binding)
        changed["native"]["binary_sha256"] = "0"*64
        with patch.object(verifier, "build_binding", return_value=changed):
            with self.assertRaisesRegex(ValueError, "current capacity build/source/runtime"):
                verifier.verify_capacity(self.root)

    def test_binding_is_checked_again_after_replay(self):
        changed = copy.deepcopy(self.binding)
        changed["python_sources"] = {}
        with patch.object(verifier, "build_binding", side_effect=[copy.deepcopy(self.binding), changed]):
            with self.assertRaisesRegex(ValueError, "binding unchanged during offline"):
                verifier.verify_capacity(self.root)

    def test_cli_refuses_overwriting_existing_frozen_report(self):
        path = self.root / "capacity-report.json"
        before = c.sha(path)
        with patch.object(verifier, "verify_capacity", return_value={"verified": True}), \
                patch.object(sys, "argv", ["verify_capacity", str(self.root), "--output", str(path)]):
            self.assertEqual(verifier.main(), 2)
        self.assertEqual(before, c.sha(path))


if __name__ == "__main__":
    unittest.main()
