"""Portable cache-episode loading and independently replayable export evidence."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from llm_infer_model.m2.spec import (
    parse_state_document, state_in_from_state_out, to_state_in_document,
)
from llm_infer_model.tensor.engine import evaluate_mapping, load_mapping
from llm_infer_model.tensor.spec import (
    InvalidInput, SpecError, build_scenario, scenario_fingerprint, workload_fingerprint,
)
from tensor_mapping import m2_cache_cli as cli


_EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "m2"
_EPISODE = "moe-cache-top1.episode.json"


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, document):
    Path(path).write_text(json.dumps(document, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8", newline="\n")


class CacheCliTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        for source in _EXAMPLES.glob("moe-cache-top1*"):
            shutil.copyfile(source, self.folder / source.name)
        self.episode_path = self.folder / _EPISODE
        self.original = _read(self.episode_path)

    def execute(self, name="output", *, states=100):
        return cli.execute_episode(self.episode_path, self.folder / name,
                                   max_candidates=100, max_expanded_states=states)

    def test_binding_unknown_fields_and_capacity_unknown_are_explicit(self):
        malformed = []
        missing_binding = copy.deepcopy(self.original)
        del missing_binding["state_binding"]
        malformed.append(missing_binding)
        extra = copy.deepcopy(self.original)
        extra["prefetch"] = True
        malformed.append(extra)
        for value in (True, False, 1.0, "128", -1):
            quota = copy.deepcopy(self.original)
            quota["expert_cache_capacity_bytes"]["dram"] = value
            malformed.append(quota)
        for document in malformed:
            with self.subTest(document=document):
                _write(self.episode_path, document)
                with self.assertRaises(InvalidInput):
                    cli.load_cache_episode(self.episode_path)
        unbounded = copy.deepcopy(self.original)
        unbounded["expert_cache_capacity_bytes"] = {"dram": None, "vram": None}
        _write(self.episode_path, unbounded)
        episode = cli.load_cache_episode(self.episode_path)
        self.assertEqual(episode.expert_cache_capacity_bytes, {"dram": None, "vram": None})

    def test_duplicate_policy_and_invocation_identity_are_refused(self):
        duplicate_policy = copy.deepcopy(self.original)
        duplicate_policy["baseline_policies"].append("lru")
        _write(self.episode_path, duplicate_policy)
        with self.assertRaisesRegex(InvalidInput, "duplicate baseline"):
            cli.load_cache_episode(self.episode_path)
        _write(self.episode_path, self.original)
        first = _read(self.folder / self.original["invocation_files"][0])
        second_path = self.folder / self.original["invocation_files"][1]
        second = _read(second_path)
        second["id"] = first["id"]
        _write(second_path, second)
        with self.assertRaisesRegex(InvalidInput, "invocation ids must be unique"):
            cli.load_cache_episode(self.episode_path)

    def test_same_layer_id_does_not_replace_immutable_document_hash(self):
        original_layer_path = self.folder / "moe-cache-top1.layer.json"
        original_layer = _read(original_layer_path)
        second_path = self.folder / self.original["invocation_files"][1]
        original_call = _read(second_path)
        alternate_path = self.folder / "alternate.layer.json"
        for change_bytes in (False, True):
            with self.subTest(change_weight_bytes=change_bytes):
                alternate = copy.deepcopy(original_layer)
                call = copy.deepcopy(original_call)
                call["layer_file"] = alternate_path.name
                if change_bytes:
                    alternate["weights"][0]["storage_bytes"] = 64
                    call["state_in"]["weight_copies"][0]["storage_bytes"] = 64
                _write(alternate_path, alternate)
                # Even semantically identical JSON has a distinct source hash.
                with alternate_path.open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write("\n")
                _write(second_path, call)
                with self.assertRaisesRegex(InvalidInput, "immutable layer document SHA256"):
                    cli.load_cache_episode(self.episode_path)

    def test_source_change_during_loading_is_refused(self):
        original_loader = cli.load_and_compile
        changed = False

        def changing_loader(path):
            nonlocal changed
            compiled = original_loader(path)
            if not changed:
                changed = True
                with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write("\n")
            return compiled

        with mock.patch.object(cli, "load_and_compile", side_effect=changing_loader):
            with self.assertRaisesRegex(InvalidInput, "sources changed during loading"):
                cli.load_cache_episode(self.episode_path)

    def test_source_change_after_search_prevents_export(self):
        original_run = cli.run_episode
        output = self.folder / "unstable-source-output"

        def changing_run(*args, **kwargs):
            result = original_run(*args, **kwargs)
            path = self.folder / self.original["invocation_files"][0]
            with path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write("\n")
            return result

        with mock.patch.object(cli, "run_episode", side_effect=changing_run):
            with self.assertRaisesRegex(InvalidInput, "inputs changed during search"):
                cli.execute_episode(self.episode_path, output,
                                    max_candidates=100, max_expanded_states=100)
        self.assertFalse(output.exists())

    def test_measured_cost_cannot_cross_the_synthetic_episode_boundary(self):
        costs_path = self.folder / "moe-cache-top1.costs.json"
        costs = _read(costs_path)
        costs["entries"][0]["provenance"] = "measured_macro"
        _write(costs_path, costs)
        with self.assertRaises(SpecError):
            cli.load_cache_episode(self.episode_path)

    def test_cli_requires_both_explicit_search_budgets(self):
        base = [str(self.episode_path), "--out", str(self.folder / "unused")]
        for extras in ([], ["--max-candidates", "100"], ["--max-expanded-states", "100"]):
            with self.subTest(extras=extras), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as refusal:
                    cli.main(base + extras)
                self.assertEqual(refusal.exception.code, 2)
        self.assertFalse((self.folder / "unused").exists())

    def test_existing_output_is_refused_without_overwriting_evidence(self):
        output = self.folder / "existing-output"
        output.mkdir()
        sentinel = output / "keep.json"
        sentinel.write_text("existing evidence\n", encoding="utf-8", newline="\n")
        with self.assertRaisesRegex(InvalidInput, "absent or empty"):
            self.execute(output.name)
        self.assertEqual(list(output.iterdir()), [sentinel])
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "existing evidence\n")

    def test_complete_cli_exports_all_baselines_and_honest_failed_cost(self):
        output = self.folder / "complete-cli"
        with contextlib.redirect_stdout(io.StringIO()):
            status = cli.main([str(self.episode_path), "--out", str(output),
                               "--max-candidates", "100", "--max-expanded-states", "100"])
        self.assertEqual(status, 0)
        summary = _read(output / "summary.json")
        self.assertEqual(summary["status"], "optimal_within_declared_space")
        self.assertEqual(summary["best_total_makespan_ns"], 62_000_000)
        self.assertTrue(summary["enumeration_complete"])
        self.assertFalse(summary["performance_calibrated"])
        self.assertFalse(summary["hardware_measurement_performed"])
        self.assertFalse(summary["online_policy_claimed"])
        self.assertFalse(summary["global_schedule_optimality_proven"])
        self.assertEqual(summary["search_information_mode"], "offline_oracle")
        self.assertFalse(summary["search_scope"]["prefetch_search"])
        self.assertFalse(summary["search_scope"]["overlap_search"])
        self.assertEqual({item["policy"] for item in summary["baselines"]},
                         set(self.original["baseline_policies"]))
        failed = next(item for item in summary["baselines"] if item["policy"] == "keep_all")
        self.assertNotEqual(failed["status"], "valid")
        self.assertIsNone(failed["total_makespan_ns"])
        self.assertTrue(failed["reason"])
        self.assertEqual(failed["calls"][-1]["error_code"], "CAPACITY_EXCEEDED")
        for baseline in summary["baselines"]:
            self.assertEqual(_read(output / "baselines" / baseline["policy"] / "candidate.json"), baseline)
        self.assertTrue((output / "best" / "candidate.json").is_file())

    def test_saved_graphs_replay_actual_state_chain_and_global_timeline(self):
        self.execute()
        output = self.folder / "output"
        candidate = _read(output / "best" / "candidate.json")
        previous = None
        offset = 0
        for index, call in enumerate(candidate["calls"]):
            folder = output / "best" / f"call-{index:03d}"
            workload = _read(folder / "workload.json")
            scenario_document = _read(folder / "scenario.json")
            scenario = build_scenario(workload, scenario_document)
            mapping = load_mapping(folder / "mapping.json", scenario)
            self.assertTrue(mapping.fingerprint_verified)
            mapping_document = _read(folder / "mapping.json")
            self.assertEqual(mapping_document["workload_fingerprint"], workload_fingerprint(scenario.workload))
            self.assertEqual(mapping_document["scenario_fingerprint"], scenario_fingerprint(scenario))
            replay = evaluate_mapping(scenario, mapping.actions)
            self.assertEqual(replay.status, "valid", replay.reason)
            self.assertEqual(replay.makespan_ns, call["makespan_ns"])
            self.assertEqual(replay.peak_dram_bytes, call["peak_dram_bytes"])
            self.assertEqual(replay.ssd_read_bytes, call["ssd_read_bytes"])
            self.assertEqual(call["start_ns"], offset)
            if previous is not None:
                expected = to_state_in_document(state_in_from_state_out(previous), layer=previous.layer)
                self.assertEqual(_read(folder / "state_in.json"), expected)
            previous = parse_state_document(_read(folder / "state_out.json")).as_state_out()
            offset += replay.makespan_ns
            for path in folder.glob("*.json"):
                self.assertNotIn(b"\r\n", path.read_bytes())
        self.assertEqual(offset, 62_000_000)
        self.assertEqual(previous.requests[0].next_position, 5)
        timeline = _read(output / "best" / "timeline.json")
        self.assertFalse(timeline["overlap_between_invocations"])
        self.assertEqual(timeline["total_makespan_ns"], offset)
        self.assertEqual(max(event["t_ns"] for event in timeline["events"]), offset)
        for index, call in enumerate(candidate["calls"]):
            events = [event for event in timeline["events"] if event["call_index"] == index]
            begin, end = call["start_ns"], call["start_ns"] + call["makespan_ns"]
            self.assertTrue(events)
            for event in events:
                self.assertLessEqual(begin, event["t_ns"])
                self.assertLessEqual(event["t_ns"], end)
                if event["start_ns"] is not None:
                    self.assertLessEqual(begin, event["start_ns"])
                    self.assertLessEqual(event["start_ns"], event["end_ns"])
                    self.assertLessEqual(event["end_ns"], end)

    def test_budget_before_first_leaf_exports_unknown_without_best(self):
        summary = self.execute("unknown-output", states=1)
        self.assertEqual(summary["status"], "unknown")
        self.assertFalse(summary["enumeration_complete"])
        self.assertFalse(summary["optimality_within_declared_space"])
        self.assertIsNone(summary["best_total_makespan_ns"])
        self.assertIsNone(summary["best_candidate_id"])
        self.assertEqual(summary["candidate_count"], 0)
        self.assertEqual(summary["termination_reason"], "max_expanded_states")
        self.assertFalse((self.folder / "unknown-output" / "best").exists())
        self.assertTrue((self.folder / "unknown-output" / "baselines" / "lru" / "candidate.json").is_file())


if __name__ == "__main__":
    unittest.main()
