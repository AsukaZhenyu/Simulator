import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from profiling.state_response.collect import (digest, make_go, verify_sources, python_sources,
    startup_retry_eligible, bind_observed_runtime_files, selected_roles, main_instrumentation)
from profiling.state_response.schedule import make_jobs


class CollectionEvidenceTests(unittest.TestCase):
    def test_absent_actor_is_a_diagnostic_control_not_an_unmeasured_idle_actor(self):
        base = make_jobs()[0]
        self.assertEqual(selected_roles(base), ["cpu-worker", "gpu-worker", "main"])
        job = dict(base, split="diagnostic", owned_roles=["main"])
        with self.assertRaisesRegex(ValueError, "without its owned actor"):
            selected_roles(job)
        job["schedule"] = dict(base["schedule"], phases=[dict(state=dict(cpu_replay=0, gpu_replay=0))])
        self.assertEqual(selected_roles(job), ["main"])
        with self.assertRaisesRegex(ValueError, "diagnostic protocol"):
            selected_roles(dict(job, split="profile"))

    def test_diagnostic_instrumentation_never_silently_becomes_a_price_source(self):
        protocol = dict(profiling_instrumentation="stamped", heldout_instrumentation="bare")
        self.assertEqual(main_instrumentation(dict(split="profile"), protocol), "stamped")
        self.assertEqual(main_instrumentation(dict(split="heldout"), protocol), "bare")
        self.assertEqual(main_instrumentation(dict(split="diagnostic", instrumentation="thread-diagnostic"), protocol),
                         "thread-diagnostic")
        with self.assertRaisesRegex(ValueError, "diagnostic job"):
            main_instrumentation(dict(split="profile", instrumentation="thread-diagnostic"), protocol)

    def test_actual_observed_runtime_paths_are_deduplicated_and_missing_stays_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            known, missing, unrelated = (Path(tmp) / name for name in
                                          ("vcomp140.dll", "ggml-missing.dll", "unrelated.dll"))
            known.write_bytes(b"fixture disk bytes; no mapped process involved")
            unrelated.write_bytes(b"not a selected runtime family")
            observations = {actor: {"modules": [{"path": str(known)}, {"path": str(missing)},
                                                  {"path": str(unrelated)}, {"path": None}]}
                            for actor in ("main", "cpu-worker")}
            files, receipt = bind_observed_runtime_files(observations)
            self.assertEqual(set(files), {str(known.resolve())})
            self.assertEqual(len(receipt["modules"]), 2)
            self.assertEqual(sum(row["on_disk_sha256"] is None for row in receipt["modules"]), 1)
            self.assertFalse(receipt["loaded_memory_bytes_verified"])

    def test_archived_python_sources_do_not_reenter_current_source_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("collect.py", "tests/test_x.py", "results/old/collect.py", "build-v3/generate.py",
                         "build/generated.py", "__pycache__/old.py"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("", encoding="utf-8")
            self.assertEqual({p.relative_to(root).as_posix() for p in python_sources(root)},
                             {"collect.py", "tests/test_x.py"})

    def test_only_pre_go_failure_with_verified_cleanup_can_be_retried(self):
        eligible = {"status": "failed", "go_issued": False,
                    "all_created_processes_exited": True, "cleanup_errors": []}
        self.assertTrue(startup_retry_eligible(eligible))
        for field, value in (("go_issued", True), ("all_created_processes_exited", False),
                             ("cleanup_errors", ["unreaped"]), ("status", "measured")):
            self.assertFalse(startup_retry_eligible(dict(eligible, **{field: value})))
        self.assertFalse(startup_retry_eligible({}))

    def test_go_is_bound_only_to_predeclared_inputs_and_external_clock(self):
        clock = SimpleNamespace(frequency_hz=10000000,
                                metadata=lambda: {"clock_id": "windows_qpc", "frequency_hz": 10000000,
                                                  "cross_process_monotonic_clock": True})
        job = make_jobs()[0]
        go, events = make_go(job["schedule"], clock, 900000000)
        self.assertEqual(go["start_tick"], 900000000)
        self.assertEqual(go["end_tick"] - go["start_tick"], 48000000)
        self.assertEqual(len(events), 4)
        self.assertEqual(go["phases"][0]["begin_tick"], go["start_tick"])
        self.assertEqual(go["phases"][-1]["end_tick"], go["end_tick"])
        self.assertTrue(all(a["end_tick"] == b["begin_tick"] for a, b in zip(events, events[1:])))
        self.assertFalse(any("elapsed_ns" in item for item in go["phases"]))

    def test_source_mutation_is_rejected_even_with_unchanged_inventory(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "native.cpp"
            path.write_bytes(b"original")
            sources = {str(path): hashlib.sha256(b"original").hexdigest()}
            manifest = {"schema": "state-response-campaign-1", "sources": sources,
                        "source_inventory_sha256": digest(sources)}
            verify_sources(manifest)
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "campaign source changed"):
                verify_sources(manifest)


if __name__ == "__main__":
    unittest.main()
