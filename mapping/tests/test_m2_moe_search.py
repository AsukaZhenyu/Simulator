"""Bounded offline MoE placement comparison and exact replay artifacts."""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from llm_infer_model.m2.expand import compile_invocation
from llm_infer_model.m2.spec import state_in_from_state_out, to_state_in_document
from llm_infer_model.tensor.engine import evaluate_mapping, load_mapping
from llm_infer_model.tensor.spec import InvalidInput, Unsupported
from tensor_mapping.m2_moe import (
    SEARCH_SPACE, enumerate_device_assignments, main, search_moe,
    summary_document, write_search_artifacts,
)
from .test_m2_moe import moe_compiled, moe_layer_document


def with_budgets(compiled, *, vram=None, cpu_scratch=None, cpu_duration=None,
                 gpu_duration=None, pcie_duration=None):
    entries = {}
    for key, cost in compiled.cost_table.entries.items():
        changes = {}
        if key[1] == "cpu":
            if cpu_scratch is not None: changes["workspace_bytes"] = cpu_scratch
            if cpu_duration is not None: changes["duration_ns"] = cpu_duration
        elif key[1] == "gpu" and gpu_duration is not None:
            changes["duration_ns"] = gpu_duration
        entries[key] = dataclasses.replace(cost, **changes)
    transfers = {key: dataclasses.replace(cost, duration_ns=pcie_duration)
                 if pcie_duration is not None and key[0] in {"dram_to_vram", "vram_to_dram"} else cost
                 for key, cost in compiled.cost_table.transfers.items()}
    table = dataclasses.replace(compiled.cost_table, entries=entries, transfers=transfers)
    invocation = compiled.invocation
    if vram is not None:
        invocation = dataclasses.replace(invocation, hardware=dataclasses.replace(invocation.hardware, vram_capacity_bytes=vram))
    return compile_invocation(compiled.layer, invocation, table)


def cost_document(compiled):
    entries = [dict(kind=kind, device=device, phase=shape.phase, batch=shape.batch,
                    tokens=shape.tokens, visible_history=shape.visible_history,
                    duration_ns=cost.duration_ns, workspace_bytes=cost.workspace_bytes,
                    provenance=cost.provenance, accounting=cost.accounting)
               for (kind, device, shape), cost in compiled.cost_table.entries.items()]
    transfers = [dict(route=route, storage_bytes=size, duration_ns=cost.duration_ns,
                      provenance=cost.provenance) for (route, size), cost in compiled.cost_table.transfers.items()]
    return dict(format_version="m2-layer-0.6", source=compiled.cost_table.source,
                entries=entries, transfer_entries=transfers)


class MoeSearchTests(unittest.TestCase):
    def test_declared_cpu_only_space_is_one_feasible_replay_with_no_gpu_invention(self):
        result = search_moe(moe_compiled(devices=("cpu",)))
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(result.declared_candidate_count, 1)
        self.assertEqual(set(dict(result.best.placements).values()), {"cpu"})
        self.assertTrue(result.enumeration_complete)
        self.assertEqual(summary_document(result)["status"], "optimal_within_declared_space")

    def test_actual_expanded_experts_and_shared_expert_are_in_assignment_space(self):
        compiled = moe_compiled(devices=("cpu", "gpu"), shared=True)
        result = search_moe(compiled, max_candidates=3)
        self.assertEqual(result.declared_candidate_count, 2 ** len(compiled.tasks))
        self.assertEqual(len(result.candidates), 3)
        self.assertEqual(set(dict(result.candidates[0].placements)), set(compiled.compute_ids))
        self.assertIn("shared_expert_ffn", {t.kind for t in compiled.tasks})
        self.assertIn("expert_ffn", {t.kind for t in compiled.tasks})
        self.assertFalse(result.enumeration_complete)

    def test_budget_exhaustion_with_feasible_candidate_does_not_claim_optimum(self):
        result = search_moe(moe_compiled(devices=("cpu", "gpu")), max_candidates=1)
        summary = summary_document(result)
        self.assertEqual(summary["status"], "feasible")
        self.assertEqual(summary["termination_reason"], "max_candidates")
        self.assertEqual(summary["candidate_count"], 1)
        self.assertFalse(summary["optimality_within_declared_space"])
        self.assertFalse(summary["global_schedule_optimality_proven"])

    def test_budget_exhaustion_with_no_feasible_candidate_is_unknown_not_infeasible(self):
        compiled = with_budgets(moe_compiled(devices=("cpu", "gpu")), vram=1, cpu_scratch=1 << 30)
        result = search_moe(compiled, max_candidates=1)
        summary = summary_document(result)
        self.assertIsNone(result.best)
        self.assertEqual(summary["status"], "unknown")
        self.assertFalse(summary["enumeration_complete"])
        self.assertFalse(summary["optimality_within_declared_space"])

    def test_exhausted_declared_space_without_feasible_plan_has_no_state(self):
        compiled = with_budgets(moe_compiled(devices=("cpu",)), cpu_scratch=1 << 30)
        result = search_moe(compiled)
        self.assertIsNone(result.best)
        self.assertEqual(summary_document(result)["status"], "infeasible_within_declared_space")
        self.assertEqual(result.ranking, ())
        self.assertIsNone(summary_document(result)["candidates"][0]["state_out"])

    def test_rejections_remain_visible_and_cannot_gain_zero_time(self):
        result = search_moe(with_budgets(moe_compiled(devices=("cpu", "gpu")), vram=1), max_candidates=4)
        self.assertTrue(result.best.is_valid)
        rejected = [c for c in summary_document(result)["candidates"] if c["status"] != "valid"]
        self.assertTrue(rejected)
        for candidate in rejected:
            self.assertEqual(candidate["error_code"], "CAPACITY_EXCEEDED")
            self.assertIsNone(candidate["makespan_ns"])
            self.assertIsNone(candidate["state_out"])
            self.assertTrue(candidate["reason"])

    def test_fast_gpu_and_expensive_pcie_change_best_declared_plan(self):
        base = with_budgets(moe_compiled(devices=("cpu", "gpu")), cpu_duration=10000, gpu_duration=1, pcie_duration=1)
        full_budget = 2 ** len(base.tasks)
        fast = search_moe(base, max_candidates=full_budget)
        self.assertTrue(fast.enumeration_complete)
        self.assertEqual(set(dict(fast.best.placements).values()), {"gpu"})
        costly = search_moe(with_budgets(base, pcie_duration=1000000), max_candidates=full_budget)
        self.assertEqual(set(dict(costly.best.placements).values()), {"cpu"})

    def test_enumeration_prefix_and_ties_are_reproducible(self):
        compiled = moe_compiled(devices=("cpu", "gpu"))
        first = enumerate_device_assignments(compiled, max_candidates=3)
        self.assertEqual(first, enumerate_device_assignments(compiled, max_candidates=3))
        self.assertEqual(set(first[0].values()), {"cpu"})
        self.assertEqual(first[1][compiled.tasks[-1].operation_id], "gpu")
        self.assertTrue(all(first[1][t.operation_id] == "cpu" for t in compiled.tasks[:-1]))

    def test_budget_must_be_explicit_positive_integer_not_bool_or_float(self):
        compiled = moe_compiled(devices=("cpu",))
        for value in (0, -1, True, 1.0, "10"):
            with self.subTest(value=value), self.assertRaises(InvalidInput):
                search_moe(compiled, max_candidates=value)

    def test_offline_search_is_explicit_even_when_replay_is_causal(self):
        compiled = moe_compiled(devices=("cpu",))
        result = search_moe(compiled)
        summary = summary_document(result)
        self.assertEqual(summary["search_information_mode"], "offline_oracle")
        self.assertEqual(summary["replay_information_mode"], compiled.invocation.information_mode)
        self.assertFalse(summary["online_policy_claimed"])
        self.assertFalse(summary["performance_calibrated"])
        self.assertEqual(summary["route_source"], compiled.invocation.route_source)
        self.assertEqual(len(summary["normalized_route_trace_sha256"]), 64)
        self.assertEqual(summary["search_space"], SEARCH_SPACE)
        for field in ("route_choice_search", "overlap_search", "prefetch_search", "eviction_search"):
            self.assertFalse(summary["search_scope"][field])

    def test_dense_version_is_refused_instead_of_reinterpreted_as_moe(self):
        compiled = moe_compiled(devices=("cpu",))
        with self.assertRaises(Unsupported):
            search_moe(dataclasses.replace(compiled, layer=dataclasses.replace(compiled.layer, format_version="m2-layer-0.5")))

    def test_prefill_best_state_continues_into_decode(self):
        prefill = search_moe(moe_compiled(devices=("cpu",)))
        state = state_in_from_state_out(prefill.best.result.state_out)
        decode = search_moe(moe_compiled(devices=("cpu",), phase="decode", state=state))
        self.assertEqual(decode.compiled.state_in, state)
        for old, new in zip(state.requests, decode.best.result.state_out.requests):
            self.assertEqual(new.next_position, old.next_position + 1)
            self.assertEqual(new.version, old.version + 1)

    def test_saved_candidates_replay_exactly_with_lf_and_persistent_state(self):
        result = search_moe(moe_compiled(devices=("cpu", "gpu")), max_candidates=3)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "results"
            write_search_artifacts(result, folder)
            for candidate in result.candidates:
                mapping = load_mapping(folder / "candidates" / candidate.id / "mapping.json", result.compiled.scenario)
                replay = evaluate_mapping(result.compiled.scenario, mapping.actions)
                self.assertTrue(mapping.fingerprint_verified)
                self.assertEqual(replay.status, candidate.status)
                self.assertEqual(replay.makespan_ns, candidate.makespan_ns)
                if candidate.result is not None:
                    self.assertEqual(replay.ssd_read_bytes, candidate.result.evaluation.ssd_read_bytes)
                    self.assertEqual(replay.h2d_bytes, candidate.result.evaluation.h2d_bytes)
            self.assertTrue((folder / "state_out.json").exists())
            for path in folder.rglob("*.json"):
                self.assertNotIn(b"\r\n", path.read_bytes())

    def test_nonempty_outputs_are_refused_without_overwriting_existing_evidence(self):
        result = search_moe(moe_compiled(devices=("cpu",)))
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            write_search_artifacts(result, folder)
            before = (folder / "summary.json").read_bytes()
            with self.assertRaises(InvalidInput):
                write_search_artifacts(result, folder)
            self.assertEqual((folder / "summary.json").read_bytes(), before)

    def test_unknown_outputs_do_not_contain_best_mapping_or_state(self):
        result = search_moe(with_budgets(moe_compiled(devices=("cpu", "gpu")), cpu_scratch=1 << 30), max_candidates=1)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            write_search_artifacts(result, folder)
            self.assertFalse((folder / "state_out.json").exists())
            self.assertFalse((folder / "best_mapping.json").exists())

    def test_cli_records_stable_source_hashes_and_raw_route_trace(self):
        compiled = moe_compiled(devices=("cpu",))
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "layer.json").write_text(json.dumps(moe_layer_document()), encoding="utf-8", newline="\n")
            (folder / "costs.json").write_text(json.dumps(cost_document(compiled)), encoding="utf-8", newline="\n")
            routes = []
            for trace in compiled.invocation.route_trace:
                entry = dataclasses.asdict(trace)
                if entry["scores"] is None: del entry["scores"]
                routes.append(entry)
            invocation = dict(format_version="m2-layer-0.6", id="moe-search-cli", layer=compiled.layer.id,
                layer_file="layer.json", costs_file="costs.json", phase=compiled.invocation.phase,
                devices=list(compiled.invocation.devices), hardware=dataclasses.asdict(compiled.invocation.hardware),
                requests=[dataclasses.asdict(r) for r in compiled.invocation.requests],
                state_in=to_state_in_document(compiled.state_in, layer=compiled.layer.id),
                route_source=compiled.invocation.route_source, information_mode=compiled.invocation.information_mode,
                route_trace=routes)
            path = folder / "invocation.json"
            path.write_text(json.dumps(invocation), encoding="utf-8", newline="\n")
            output = folder / "search"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main([str(path), "--out", str(output), "--max-candidates", "1"]), 0)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            for role in ("invocation", "layer", "costs"):
                self.assertEqual(summary["source_inputs"][role]["sha256"], hashlib.sha256((folder / (role + ".json")).read_bytes()).hexdigest())
            canonical = json.dumps(routes, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.assertEqual(summary["source_inputs"]["inline_route_trace"]["canonical_sha256"], hashlib.sha256(canonical).hexdigest())
            self.assertFalse(summary["online_policy_claimed"])

    def test_exact_budget_exhausts_space_but_one_fewer_does_not(self):
        compiled = moe_compiled(devices=("cpu", "gpu"))
        count = 2 ** len(compiled.tasks)
        exact = search_moe(compiled, max_candidates=count)
        short = search_moe(compiled, max_candidates=count - 1)
        self.assertEqual(len(exact.candidates), count)
        self.assertEqual(len(short.candidates), count - 1)
        self.assertTrue(exact.enumeration_complete)
        self.assertFalse(short.enumeration_complete)
        self.assertTrue(summary_document(exact)["optimality_within_declared_space"])
        self.assertFalse(summary_document(short)["optimality_within_declared_space"])

    def test_cached_decode_avoids_rereading_expert_weights_from_ssd(self):
        prefill = search_moe(moe_compiled(devices=("cpu",)))
        self.assertGreater(prefill.best.result.evaluation.ssd_read_bytes, 0)
        state = state_in_from_state_out(prefill.best.result.state_out)
        decode = search_moe(moe_compiled(devices=("cpu",), phase="decode", state=state))
        self.assertEqual(decode.best.result.evaluation.ssd_read_bytes, 0)
        self.assertEqual(decode.best.result.state_out.weight_copies, prefill.best.result.state_out.weight_copies)

    def test_cli_changed_source_hash_refuses_search_and_creates_no_artifacts(self):
        compiled = moe_compiled(devices=("cpu",))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "results"
            with mock.patch("tensor_mapping.m2_moe._source_snapshot_moe", side_effect=[{"sha256": "first"}, {"sha256": "changed"}]), \
                 mock.patch("tensor_mapping.m2_moe.load_and_compile", return_value=compiled), \
                 mock.patch("tensor_mapping.m2_moe.search_moe") as search, \
                 contextlib.redirect_stderr(io.StringIO()):
                code = main([str(Path(temporary) / "invocation.json"), "--out", str(output)])
            self.assertEqual(code, 1)
            search.assert_not_called()
            self.assertFalse(output.exists())

    def test_cli_missing_input_is_refused_without_creating_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            with contextlib.redirect_stderr(io.StringIO()):
                code = main([str(Path(temporary) / "missing.json"), "--out", str(output)])
            self.assertEqual(code, 1)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
