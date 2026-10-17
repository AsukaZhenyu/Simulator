"""Strict schedule inputs and independently replayable overlap CLI exports."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from llm_infer_model.tensor.engine import evaluate_mapping, load_mapping
from llm_infer_model.tensor.spec import InvalidInput, build_scenario
from tensor_mapping import m2_overlap_cli as cli


EXAMPLES = Path(__file__).resolve().parents[1] / "examples/m2"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


class OverlapCliTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        for source in EXAMPLES.glob("moe-cache-top1*"):
            shutil.copyfile(source, self.folder / source.name)
        self.episode_path = self.folder / "schedule.json"
        self.document = dict(format_version=cli.FORMAT, id="portable-overlap-two-calls",
            invocation_files=["moe-cache-top1-call-0.json", "moe-cache-top1-call-1.json"],
            state_binding=cli.STATE_BINDING, expert_cache_capacity_bytes={"dram": 128, "vram": 128},
            placement_mode="fixed_first_declared", prefetch_depth=1, overlap=True,
            baseline_policies=["serial_lru", "demand_overlap_lru", "prefetch_lru"])
        write(self.episode_path, self.document)

    def execute(self, name="output", *, expanded=1000, generated=5000):
        return cli.execute_episode(self.episode_path, self.folder / name,
            max_expanded_states=expanded, max_generated_states=generated, max_baseline_actions=1000)

    def test_schema_never_guesses_prefetch_overlap_or_binding(self):
        for key in ("prefetch_depth", "overlap", "state_binding"):
            malformed = copy.deepcopy(self.document)
            del malformed[key]
            write(self.episode_path, malformed)
            with self.assertRaises(InvalidInput):
                cli.episode_raw(self.episode_path)
        for key, values in (("prefetch_depth", (True, -1, 1.0, "1")), ("overlap", (1, None, "true"))):
            for value in values:
                malformed = copy.deepcopy(self.document)
                malformed[key] = value
                write(self.episode_path, malformed)
                with self.assertRaises(InvalidInput):
                    cli.episode_raw(self.episode_path)
        malformed = copy.deepcopy(self.document)
        malformed["arbitrary_prediction"] = True
        write(self.episode_path, malformed)
        with self.assertRaises(InvalidInput):
            cli.episode_raw(self.episode_path)

    def test_baselines_and_quota_are_explicit(self):
        for values in (("baseline_policies", []), ("baseline_policies", ["prefetch_lru", "prefetch_lru"]),
                       ("baseline_policies", ["rl"]), ("expert_cache_capacity_bytes", {"dram": True, "vram": 128})):
            malformed = copy.deepcopy(self.document)
            malformed[values[0]] = values[1]
            write(self.episode_path, malformed)
            with self.assertRaises(InvalidInput):
                cli.episode_raw(self.episode_path)

    def test_same_layer_id_cannot_hide_a_different_document(self):
        layer = self.folder / "moe-cache-top1.layer.json"
        changed_layer = self.folder / "same-id.layer.json"
        changed_layer.write_bytes(layer.read_bytes() + b"\n")
        call = self.folder / "moe-cache-top1-call-1.json"
        document = read(call)
        document["layer_file"] = changed_layer.name
        write(call, document)
        with self.assertRaisesRegex(InvalidInput, "immutable layer"):
            cli.load_episode(self.episode_path)

    def test_output_protection_precedes_any_search(self):
        output = self.folder / "existing"
        output.mkdir()
        (output / "valuable.txt").write_text("keep", encoding="utf-8")
        with mock.patch.object(cli, "load_episode") as load:
            with self.assertRaisesRegex(InvalidInput, "absent or empty"):
                self.execute("existing")
            load.assert_not_called()
        self.assertEqual((output / "valuable.txt").read_text(), "keep")

    def test_budgets_cannot_be_unknown_or_boolean(self):
        for key in ("max_expanded_states", "max_generated_states", "max_baseline_actions"):
            for value in (0, True, None):
                args = dict(max_expanded_states=100, max_generated_states=100, max_baseline_actions=100)
                args[key] = value
                with self.assertRaises(InvalidInput):
                    cli.execute_episode(self.episode_path, self.folder / "invalid", **args)

    def test_engine_interval_metrics_union_and_exclusive_resource(self):
        events = [SimpleNamespace(resource="cpu_compute", start_ns=0, end_ns=10),
                  SimpleNamespace(resource="ssd_io", start_ns=2, end_ns=5),
                  SimpleNamespace(resource="pcie_copy", start_ns=4, end_ns=8),
                  SimpleNamespace(resource="gpu_compute", start_ns=8, end_ns=12)]
        metrics = cli.overlap_metrics(SimpleNamespace(events=events))
        self.assertEqual(metrics["compute_transfer_overlap_ns"], 6)
        self.assertEqual(metrics["cpu_gpu_overlap_ns"], 2)
        self.assertEqual(metrics["ssd_pcie_overlap_ns"], 1)
        self.assertEqual(metrics["resource_busy_ns"]["cpu_compute"], 10)
        events.append(SimpleNamespace(resource="ssd_io", start_ns=4, end_ns=7))
        with self.assertRaisesRegex(InvalidInput, "exclusive resource"):
            cli.overlap_metrics(SimpleNamespace(events=events))

    def test_exports_replay_saved_graph_and_actual_state_chain(self):
        summary = self.execute()
        self.assertIsNotNone(summary["best_total_makespan_ns"])
        self.assertFalse(summary["performance_calibrated"])
        self.assertFalse(summary["online_policy_claimed"])
        output = self.folder / "output/best"
        previous = None
        offset = 0
        for index, saved in enumerate(summary["best"]["calls"]):
            folder = output / f"call-{index:03d}"
            if previous is not None:
                self.assertEqual(read(folder / "state_in.json"), {key: previous[key]
                    for key in ("requests", "weight_copies", "layer")})
            scenario = build_scenario(read(folder / "workload.json"), read(folder / "scenario.json"))
            mapping = load_mapping(folder / "mapping.json", scenario)
            replay = evaluate_mapping(scenario, mapping.actions)
            self.assertTrue(mapping.fingerprint_verified)
            self.assertEqual(replay.status, "valid")
            self.assertEqual(replay.makespan_ns, saved["makespan_ns"])
            self.assertEqual(cli.overlap_metrics(replay), saved["overlap_metrics"])
            self.assertEqual(saved["start_ns"], offset)
            offset += replay.makespan_ns
            previous = read(folder / "state_out.json")
        self.assertEqual(offset, summary["best_total_makespan_ns"])
        self.assertFalse(read(output / "timeline.json")["overlap_between_invocations"])

    def test_search_cutoff_retains_verified_seed_without_claiming_optimality(self):
        summary = self.execute(expanded=1, generated=2)
        self.assertEqual(summary["status"], "feasible")
        self.assertIsNotNone(summary["best"])
        self.assertFalse(summary["optimality_within_declared_space"])
        self.assertLessEqual(summary["expanded_states"], 1)
        self.assertLessEqual(summary["generated_states"], 2)

    def test_explicit_compressed_states_are_lossless_and_indexed(self):
        summary = cli.execute_episode(self.episode_path, self.folder / "compressed",
            max_expanded_states=1, max_generated_states=2, max_baseline_actions=1000,
            state_artifact_format="json-gzip")
        self.assertEqual(summary["state_artifact_format"], "json-gzip")
        folder = self.folder / "compressed/best"
        index = read(folder / "state-artifacts.json")
        self.assertEqual(len(index["artifacts"]), 2)
        for artifact in index["artifacts"]:
            path = folder / artifact["path"]
            raw = gzip.decompress(path.read_bytes())
            self.assertEqual(len(raw), artifact["uncompressed_size_bytes"])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), artifact["uncompressed_sha256"])
            self.assertIn("states", json.loads(raw))
            self.assertFalse(path.with_suffix("").exists())

    def test_source_changes_during_search_are_refused(self):
        from tensor_mapping import m2_overlap_search
        original = m2_overlap_search.search_schedule_episode
        def altered(*args, **kwargs):
            result = original(*args, **kwargs)
            self.episode_path.write_bytes(self.episode_path.read_bytes() + b"\n")
            return result
        with mock.patch.object(m2_overlap_search, "search_schedule_episode", side_effect=altered):
            with self.assertRaisesRegex(InvalidInput, "sources changed during search"):
                self.execute(expanded=1, generated=2)

    def test_gpu_seed_is_explicit_and_revalidated_in_joint_space(self):
        self.document["baseline_policies"] = ["serial_lru", "prefetch_gpu"]
        write(self.episode_path, self.document)
        with self.assertRaisesRegex(InvalidInput, "enumerate"):
            cli.load_episode(self.episode_path)
        self.document["placement_mode"] = "enumerate"
        write(self.episode_path, self.document)
        summary = self.execute(expanded=1, generated=2)
        self.assertEqual(summary["status"], "feasible")
        gpu = next(item for item in summary["baselines"] if item["strategy_label"] == "prefetch_gpu")
        self.assertEqual(gpu["status"], "valid")
        self.assertTrue(all(set(call["placements"].values()) == {"gpu"} for call in gpu["calls"]))
        self.assertEqual(summary["best_total_makespan_ns"], gpu["total_makespan_ns"])
        self.assertFalse(summary["optimality_within_declared_space"])
