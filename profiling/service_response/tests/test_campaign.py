"""Predeclared controller trajectories, without hardware or timing tests."""
import unittest
import argparse
from contextlib import nullcontext
from pathlib import Path
import tempfile
from unittest.mock import patch

from profiling.service_response.campaign import declared_protocol, recovery_schedule, prepare, collect
from profiling.state_calibration import run
from profiling.service_response.calibration import validate_protocol
from profiling.state_response.collect import native_evidence_bindings


class CampaignTests(unittest.TestCase):
    def test_declared_protocol_is_the_exact_model_contract(self):
        protocol = declared_protocol()
        self.assertEqual(validate_protocol(protocol), protocol)
        self.assertFalse(protocol["heldout_refit"])
        self.assertEqual(protocol["combined_state_role"], "validation_only")

    def test_profile_includes_short_pulses_and_long_isolated_prices(self):
        schedule = recovery_schedule(split="profile", seed=18, settle_ms=1000)
        phases = schedule["phases"]
        self.assertEqual(phases[0]["state"]["cpu_replay"], 0)
        self.assertEqual(phases[0]["state"]["gpu_replay"], 0)
        for key in ("cpu_replay", "gpu_replay"):
            self.assertTrue(any(p["state"][key] == 1 and p["end_ns"] - p["begin_ns"] == 100000000 for p in phases))
        self.assertEqual(sum(p["end_ns"] - p["calibration_begin_ns"] > 1 for p in phases), 13)
        self.assertEqual(schedule["duration_ns"], 23600000000)

    def test_heldout_changes_numeric_seed_order_and_durations(self):
        a = recovery_schedule(split="heldout", seed=100, settle_ms=1000)
        b = recovery_schedule(split="heldout", seed=101, settle_ms=1000)
        self.assertNotEqual(a["phases"], b["phases"])
        self.assertEqual(a["phases"][-1]["state"], a["phases"][0]["state"])
        self.assertTrue(all(p["calibration_begin_ns"] == p["begin_ns"] for p in a["phases"]))
        self.assertTrue(all(p["state"]["host_memory_bytes"] == p["state"]["device_memory_bytes"] == 0 for p in a["phases"]))

    def test_invalid_split_rejects(self):
        with self.assertRaises(ValueError):
            recovery_schedule(split="train_and_test", seed=1, settle_ms=1000)

    def test_compact_source_receipt_does_not_fabricate_action_prices(self):
        bindings = dict(program="program.json", native="native.json", events="events.json", numerical="numeric.json")
        sources = {name: "a" * 64 for name in bindings.values()}
        kwargs = dict(program={"program_sha256": "b" * 64}, context={"scope": "fixture"},
                      run_id="profile-fixture", split="profile", seed=1, process_id=123,
                      source_files=sources, numerical_verification={"passed": True}, raw_bindings=bindings)
        receipt = native_evidence_bindings(**kwargs)
        self.assertEqual(receipt["source_files"], sources)
        self.assertEqual(receipt["raw_bindings"], bindings)
        self.assertFalse(receipt["action_intervals_expanded"])
        self.assertFalse(receipt["timing_parameters_estimated"])
        with self.assertRaises(ValueError):
            native_evidence_bindings(**dict(kwargs, numerical_verification={"passed": False}))
        with self.assertRaises(ValueError):
            native_evidence_bindings(**dict(kwargs, source_files={}))

    def test_prepare_preserves_exclusive_base_manifest_then_writes_final_once(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "campaign"
            args = argparse.Namespace(out=out, binary=Path("probe.exe"), runtime_dir=[Path("runtime")],
                cuda_runtime=Path("cudart.dll"), seed=1, settle_ms=1000, max_step_ns=250000,
                main_affinity_mask=5, main_thread_affinity_mask=4,
                cpu_worker_affinity_mask=5, gpu_worker_affinity_mask=48)
            base = dict(protocol={}, context={}, sources={}, jobs=[dict(id="p", split="profile", schedule={"seed":1})])
            def setup(unused):
                out.mkdir()
                run.save(out / "manifest.json", base)
            with patch("profiling.service_response.campaign.common.prepare", side_effect=setup), \
                 patch("profiling.service_response.campaign.common.verify_sources"), \
                 patch("profiling.service_response.campaign.common.python_sources", return_value=[]):
                result = prepare(args)
            self.assertEqual(run.read(out / "collector-base-manifest.json"), base)
            self.assertIn("service_response", run.read(out / "manifest.json")["protocol"])
            self.assertFalse(result["hardware_executed"])
            self.assertTrue((out / "recorded-sources/archive.json").exists())

    def test_three_job_checkpoints_and_final_outputs_never_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            run.save(out / "manifest.json", {"fixture": True})
            jobs = [dict(id="p"+str(i),split="profile") for i in range(3)]
            manifest = dict(jobs=jobs, protocol=dict(service_response=declared_protocol(),maximum_pre_go_startup_retries=0))
            def execute(unused_out,unused_manifest,job,**kwargs):
                self.assertTrue(kwargs["native_evidence_only"])
                return dict(status="measured",process_id=100+int(job["id"][-1]),job=job)
            with patch("profiling.service_response.campaign.common.load_manifest",return_value=manifest), \
                 patch("profiling.service_response.campaign.common.verify_sources"), \
                 patch("profiling.service_response.campaign.run.measurement_lock",side_effect=nullcontext), \
                 patch("profiling.service_response.campaign.common.execute_job",side_effect=execute) as executor, \
                 patch("profiling.service_response.analysis.extract_job",side_effect=lambda path,row,**kw: {"run_id":row["job"]["id"]}):
                args=argparse.Namespace(out=out,split="profile",frozen=None)
                self.assertEqual(collect(args)["independent_processes"],3)
                self.assertEqual(len(run.read(out / "profile-records.json")),3)
                self.assertEqual(len(list(out.glob("profile-records-checkpoint-*.json"))),3)
                with self.assertRaises(ValueError): collect(args)
                self.assertEqual(executor.call_count,3)


if __name__ == "__main__":
    unittest.main()
