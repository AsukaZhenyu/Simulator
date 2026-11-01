"""Collector protocol checks with fake actors; no native program is launched."""
from argparse import Namespace
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
for directory in (ROOT, ROOT / "modeling", ROOT / "profiling"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from profiling.available_service import campaign


class PhysicalCampaignTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="physical-campaign-", dir=ROOT / "build")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def prepare_fixture(self):
        binaries = []
        for name in ("physical", "background"):
            path = self.directory / name / (name + ".exe")
            path.parent.mkdir()
            path.write_bytes(("fake-not-executable-" + name).encode())
            campaign.run.save(path.parent / "build-manifest.json", dict(returncode=0,
                binary_sha256=campaign.run.sha(path), bound_files={}))
            binaries.append(path)
        virtual_root = self.directory / "workspace"
        source = virtual_root / "modeling/llm_infer_model/fixture.py"
        source.parent.mkdir(parents=True)
        source.write_text("# synthetic protocol source\n", encoding="utf-8", newline="\n")
        args = Namespace(out=self.directory / "campaign", binary=binaries[0], background_binary=binaries[1],
                         runtime_dir=[], seed=12345)
        case = SimpleNamespace(compile=lambda: object())
        def export(_compiled, seed):
            return dict(seed=seed, program_sha256=campaign.digest(dict(seed=seed)))
        with patch.object(campaign, "ROOT", virtual_root), \
             patch("profiling.highlevel_accuracy.cases.build_case", return_value=case) as build, \
             patch("profiling.highlevel_accuracy.run.case_specs", return_value=[dict(
                 id="h128-q16-history0-attention-ffn-cut128", kwargs={"threads": 2})]), \
             patch("profiling.highlevel_accuracy.bridge.export_plan", side_effect=export), \
             patch("profiling.highlevel_accuracy.bridge.reference_ports", return_value={"fixture": True}), \
             patch("profiling.state_response.collect.hardware_identity", return_value={"synthetic": True}):
            outcome = campaign.prepare(args)
        return args, campaign.run.read(args.out / "manifest.json"), outcome, build

    def test_preparation_declares_disjoint_numeric_seeds_and_all_conditions(self):
        args, manifest, outcome, build = self.prepare_fixture()
        self.assertFalse(outcome["native_timed_work_started"])
        self.assertEqual(outcome["jobs"], 34)
        self.assertEqual(build.call_args.kwargs["threads"], 1)
        profile = [v["seed"] for k, v in manifest["programs"].items() if k.startswith("profile")]
        heldout = [v["seed"] for k, v in manifest["programs"].items() if k.startswith("heldout")]
        self.assertEqual(len(set(profile)), 3)
        self.assertEqual(len(set(heldout)), 3)
        self.assertTrue(set(profile).isdisjoint(heldout))
        for split in ("profile", "heldout"):
            jobs = [j for j in manifest["jobs"] if j["split"] == split]
            self.assertEqual(len(jobs), 12)
            for repeat in range(3):
                selected = [j for j in jobs if j["program_key"] == f"{split}-{repeat}"]
                self.assertEqual({j["condition"] for j in selected}, {"idle", "cpu", "gpu", "both"})
                self.assertTrue(all(j["references"] and j["instrumentation"] == "cupti" for j in selected))
        campaign.verify(manifest)
        index = campaign.run.read(args.out / "recorded-sources/archive.json")
        self.assertEqual(index["manifest_sha256"], campaign.run.sha(args.out / "manifest.json"))

    def test_preparation_is_write_once_and_source_drift_is_rejected(self):
        args, manifest, _, _ = self.prepare_fixture()
        with self.assertRaises(FileExistsError):
            campaign.prepare(args)
        source = Path(next(name for name in manifest["sources"] if name.endswith("fixture.py")))
        source.write_text("# changed\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "bound source changed"):
            campaign.verify(manifest)

    def test_smoke_actor_receives_matching_raw_qpc_go_and_cleanup(self):
        args, manifest, _, _ = self.prepare_fixture()
        actors, pins, go_records = [], [], []
        class Clock:
            frequency_hz = 1000
            tick = 10_000
            def now(self):
                self.tick += 1
                return self.tick
            def metadata(self):
                return {"clock_id": "windows_qpc", "frequency_hz": self.frequency_hz}
        class Actor:
            def __init__(self, name, directory, command, env, clock, **kwargs):
                self.name, self.directory, self.command = name, directory, command
                directory.mkdir(parents=True)
                self.process = SimpleNamespace(pid=300 + len(actors))
                self.record = {"ready": {"main_thread_id": 500 + len(actors)}}
                actors.append(self)
            def stop(self, cleanup=False):
                campaign.run.save(self.directory / "result.json", dict(schema="physical-stage-native-1",
                    verification_native={}, instrumentation="cupti", cupti={"all_requested_records_retained": True,
                    "cupti_dropped_records": 0, "callback_errors": [], "records": []}))
                return dict(name=self.name, pid=self.process.pid, returncode=0,
                            process_exited_verified=True, forced_termination=False)
        class Pin:
            def __init__(self):
                self.receipt = {"fixture": True}
                self.observed, self.closed = False, False
                pins.append(self)
            def observe(self):
                self.observed = True
            def close(self):
                self.closed = True
        def complete(child, clock, timeout):
            go_records.append(campaign.run.read(child.directory / "go.json"))
            return {"process_id": child.process.pid, "completed_samples": 2}
        job = next(j for j in manifest["jobs"] if j["split"] == "smoke")
        with patch.object(campaign, "NativeClock", Clock), \
             patch.object(campaign, "OwnedChild", Actor), \
             patch.object(campaign.OwnedActor, "from_popen", return_value=object()), \
             patch.object(campaign, "pin_owned_main_thread", side_effect=lambda *a, **kw: Pin()), \
             patch.object(campaign, "wait_complete", side_effect=complete), \
             patch.object(campaign.run, "child_environment", return_value={}), \
             patch("profiling.highlevel_accuracy.bridge.verify_native", return_value={}):
            outcome = campaign.execute(args.out, manifest, job)
        self.assertEqual(outcome["status"], "completed")
        self.assertTrue(outcome["cleanup_verified"])
        self.assertEqual(len(actors), 1)
        self.assertEqual(len(actors[0].command), 12)
        self.assertEqual(actors[0].command[2:8], ["full", "cupti", "none", "20", "2", "0"])
        self.assertEqual(go_records[0]["schema"], "physical-stages-go-1")
        self.assertEqual(go_records[0]["clock_id"], "windows_qpc")
        self.assertEqual(go_records[0]["frequency_hz"], 1000)
        self.assertGreater(go_records[0]["end_tick"], go_records[0]["start_tick"])
        self.assertTrue(all(p.closed and p.observed for p in pins))
        self.assertFalse(outcome.get("affinity_observation_errors"))


if __name__ == "__main__":
    unittest.main()
