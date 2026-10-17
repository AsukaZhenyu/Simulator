"""Restricted hybrid search replays real shared-kernel plans without hardware."""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from llm_infer_model.m2.costs import parse_cost_table
from llm_infer_model.m2.expand import compile_invocation
from llm_infer_model.m2.runner import run_invocation
from llm_infer_model.m2.spec import load_state_document, state_in_from_state_out, to_state_in_document
from llm_infer_model.tensor.engine import ACTION_COMPUTE, evaluate_mapping, load_mapping
from llm_infer_model.tensor.spec import InvalidInput, Unsupported
from tensor_mapping.m2_hybrid import (
    SEARCH_SPACE, enumerate_device_assignments, main, search_hybrid,
    summary_document, write_search_artifacts,
)

from .test_m2_hybrid import (
    hybrid_compiled, hybrid_cost_document, hybrid_invocation, hybrid_layer_document,
)


def with_costs(compiled, *, transfer_duration=None, attention_gpu_duration=None, cpu_scratch=None):
    raw = hybrid_cost_document(compiled.layer, compiled.invocation)
    if transfer_duration is not None:
        for entry in raw["transfer_entries"]:
            entry["duration_ns"] = transfer_duration
    for entry in raw["entries"]:
        if attention_gpu_duration is not None and entry["kind"] == "attention" and entry["device"] == "gpu":
            entry["duration_ns"] = attention_gpu_duration
        if cpu_scratch is not None and entry["device"] == "cpu":
            entry["workspace_bytes"] = cpu_scratch
    return compile_invocation(compiled.layer, compiled.invocation, parse_cost_table(raw))


def with_vram(compiled, capacity):
    invocation = dataclasses.replace(compiled.invocation, hardware=dataclasses.replace(
        compiled.invocation.hardware, vram_capacity_bytes=capacity))
    return compile_invocation(compiled.layer, invocation, compiled.cost_table)


class HybridSearchTests(unittest.TestCase):
    def test_all_eight_assignments_are_compared_and_mixed_plan_wins(self):
        compiled = hybrid_compiled()
        result = search_hybrid(compiled)
        self.assertEqual(len(result.candidates), 8)
        self.assertEqual(len({candidate.placements for candidate in result.candidates}), 8)
        self.assertTrue(all(candidate.is_valid for candidate in result.candidates))
        self.assertEqual(tuple(device for _, device in result.best.placements), ("cpu", "gpu", "cpu"))
        self.assertEqual(result.best.makespan_ns, 3090)
        self.assertEqual(result.candidates[0].makespan_ns, 11000)
        self.assertEqual(result.candidates[-1].makespan_ns, 21150)
        self.assertEqual([action.device for action in result.best.actions if action.kind == ACTION_COMPUTE],
                         ["cpu", "gpu", "cpu"])

    def test_transfer_cost_changes_best_plan_instead_of_hardcoding_mixed(self):
        result = search_hybrid(with_costs(hybrid_compiled(), transfer_duration=10000))
        self.assertEqual(tuple(device for _, device in result.best.placements), ("cpu", "cpu", "cpu"))
        self.assertEqual(result.best.makespan_ns, 11000)
        self.assertEqual(result.best.result.evaluation.h2d_bytes, 0)
        self.assertEqual(result.best.result.evaluation.d2h_bytes, 0)

    def test_low_vram_retains_seven_capacity_rejections_and_cpu_plan(self):
        result = search_hybrid(with_vram(hybrid_compiled(), 1))
        self.assertEqual(len(result.candidates), 8)
        self.assertEqual(len(result.ranking), 1)
        self.assertEqual(result.best.index, 0)
        for candidate in result.candidates[1:]:
            document = summary_document(result)["candidates"][candidate.index]
            self.assertEqual(document["status"], "invalid_mapping")
            self.assertEqual(document["error_code"], "CAPACITY_EXCEEDED")
            self.assertIsNotNone(document["error_action_index"])
            self.assertIsNone(document["makespan_ns"])
            self.assertIsNone(document["state_out"])
            self.assertTrue(document["reason"])

    def test_declared_cpu_only_space_does_not_gain_gpu_candidates(self):
        compiled = hybrid_compiled(devices=("cpu",))
        assignments = enumerate_device_assignments(compiled)
        self.assertEqual(len(assignments), 1)
        self.assertEqual(set(assignments[0].values()), {"cpu"})
        result = search_hybrid(compiled)
        self.assertEqual(result.best.makespan_ns, 11000)

    def test_equal_makespans_use_stable_enumeration_order(self):
        result = search_hybrid(with_costs(hybrid_compiled(), attention_gpu_duration=8910))
        self.assertEqual(result.candidates[0].makespan_ns, result.candidates[2].makespan_ns)
        self.assertEqual([candidate.index for candidate in result.ranking[:2]], [0, 2])
        self.assertEqual(result.best.index, 0)

    def test_no_feasible_serial_plan_has_no_best_or_state(self):
        compiled = with_costs(with_vram(hybrid_compiled(), 1), cpu_scratch=1 << 21)
        result = search_hybrid(compiled)
        self.assertIsNone(result.best)
        self.assertEqual(result.ranking, ())
        self.assertEqual(summary_document(result)["status"], "infeasible_within_declared_space")
        self.assertFalse(summary_document(result)["global_schedule_optimality_proven"])

    def test_scope_is_explicit_and_does_not_claim_full_action_space_optimum(self):
        summary = summary_document(search_hybrid(hybrid_compiled()))
        self.assertEqual(summary["search_space"], SEARCH_SPACE)
        self.assertTrue(summary["optimality_within_declared_space"])
        self.assertFalse(summary["global_schedule_optimality_proven"])
        for field in ("transfer_order_search", "prefetch_search", "overlap_search", "eviction_search", "recomputation_search"):
            self.assertFalse(summary["search_scope"][field])
        with self.assertRaises(Unsupported):
            search_hybrid(dataclasses.replace(hybrid_compiled(), layer=dataclasses.replace(
                hybrid_compiled().layer, format_version="m2-layer-0.4")))

    def test_selected_state_is_reusable_by_the_next_decode_invocation(self):
        prefill = search_hybrid(hybrid_compiled())
        state = state_in_from_state_out(prefill.best.result.state_out)
        decode = search_hybrid(hybrid_compiled(state=state, phase="decode"))
        self.assertEqual(decode.best.result.state_out.requests[0].next_position, 4)
        self.assertEqual(decode.best.result.state_out.requests[0].version, 3)
        self.assertEqual(len(decode.best.result.state_out.requests[0].blocks), 4)
        self.assertEqual(decode.compiled.state_in, state)

    def test_candidate_artifacts_replay_and_preserve_metrics_with_lf(self):
        result = search_hybrid(hybrid_compiled())
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "nested" / "output"
            write_search_artifacts(result, folder)
            summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["candidate_count"], 8)
            for candidate in result.candidates:
                path = folder / "candidates" / candidate.id / "mapping.json"
                loaded = load_mapping(path, result.compiled.scenario)
                self.assertTrue(loaded.fingerprint_verified)
                replay = evaluate_mapping(result.compiled.scenario, loaded.actions)
                self.assertEqual(replay.status, candidate.status)
                self.assertEqual(replay.makespan_ns, candidate.makespan_ns)
                self.assertEqual(replay.h2d_bytes, candidate.result.evaluation.h2d_bytes)
                self.assertEqual(replay.d2h_bytes, candidate.result.evaluation.d2h_bytes)
            best = load_mapping(folder / "best_mapping.json", result.compiled.scenario)
            self.assertEqual(run_invocation(result.compiled, actions=best.actions).state_out,
                             result.best.result.state_out)
            self.assertEqual(load_state_document(folder / "state_out.json").as_state_out(),
                             result.best.result.state_out)
            for path in folder.rglob("*.json"):
                self.assertNotIn(b"\r\n", path.read_bytes(), path.name)

    def test_nonempty_output_is_refused_without_stale_candidates_or_overwrites(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            write_search_artifacts(search_hybrid(hybrid_compiled()), folder)
            original = {path.relative_to(folder): path.read_bytes() for path in folder.rglob("*.json")}
            with self.assertRaisesRegex(InvalidInput, "absent or empty"):
                write_search_artifacts(search_hybrid(hybrid_compiled(devices=("cpu",))), folder)
            self.assertEqual({path.relative_to(folder): path.read_bytes() for path in folder.rglob("*.json")}, original)

    def test_infeasible_artifacts_contain_no_best_mapping_or_persistent_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            failed = search_hybrid(with_costs(with_vram(hybrid_compiled(), 1), cpu_scratch=1 << 21))
            write_search_artifacts(failed, folder)
            self.assertFalse((folder / "best_mapping.json").exists())
            self.assertFalse((folder / "state_out.json").exists())
            self.assertIsNone(json.loads((folder / "summary.json").read_text(encoding="utf-8"))["best_candidate_id"])

    def test_cli_loads_relative_files_and_writes_search_outputs(self):
        compiled = hybrid_compiled()
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            raw = hybrid_layer_document()
            (folder / "layer.json").write_text(json.dumps(raw), encoding="utf-8", newline="\n")
            (folder / "costs.json").write_text(json.dumps(hybrid_cost_document(compiled.layer, compiled.invocation)),
                                               encoding="utf-8", newline="\n")
            invocation = dict(format_version=compiled.invocation.format_version, id="cli-hybrid",
                layer=compiled.layer.id, layer_file="layer.json", costs_file="costs.json",
                phase=compiled.invocation.phase, devices=list(compiled.invocation.devices),
                hardware=dataclasses.asdict(compiled.invocation.hardware),
                requests=[dataclasses.asdict(request) for request in compiled.invocation.requests],
                state_in=to_state_in_document(compiled.state_in, layer=compiled.layer.id))
            path = folder / "invocation.json"
            path.write_text(json.dumps(invocation), encoding="utf-8", newline="\n")
            output = folder / "search-output"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main([str(path), "--out", str(output)]), 0)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["valid_candidate_count"], 8)
            self.assertFalse(summary["hardware_measurement_performed"])
            self.assertEqual(summary["cost_provenance"], ["synthetic_layer", "synthetic_transfer"])
            for role, filename in (("invocation", "invocation.json"), ("layer", "layer.json"), ("costs", "costs.json")):
                self.assertEqual(summary["source_inputs"][role]["sha256"], hashlib.sha256((folder / filename).read_bytes()).hexdigest())
            self.assertTrue((output / "workload.json").exists())
            self.assertTrue((output / "scenario.json").exists())

    def test_cli_missing_input_is_refused_without_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            with contextlib.redirect_stderr(io.StringIO()):
                code = main([str(Path(temporary) / "missing.json"), "--out", str(output)])
            self.assertEqual(code, 1)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
