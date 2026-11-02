"""Offline providers only: no process launches, native queries or GPU work."""
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from profiling.state_response import worker_activity as w
from profiling.resource_calibration import counters


class Clock:
    frequency_hz = 10_000_000
    def __init__(self):
        self.tick = 0
    def now(self):
        self.tick += 1
        return self.tick
    def metadata(self):
        return dict(clock_id="fixture-QPC", frequency_hz=self.frequency_hz)


def snapshot(begin, end, user, *, identity=True):
    return dict(schema=w.SCHEMA, actor_id="owned-cpu", process_id=99,
        process_identity=dict(status="verified" if identity else "unknown"),
        cpu_counter_snapshot=dict(schema=counters.SCHEMA, clock=Clock().metadata(),
            query_begin_tick=begin, query_end_tick=end,
            process=dict(handle_identity=123, cpu_counter_begin_tick=begin, cpu_counter_end_tick=end,
                values=dict(creation_time_100ns=1, kernel_time_100ns=0, user_time_100ns=user))))


class WorkerActivityTests(unittest.TestCase):
    def setUp(self):
        self.actor = w.OwnedActor("owned-cpu", 99, 123)

    def test_launch_environment_preserves_raw_case_spacing_empty_and_extra_keys(self):
        result = w.capture_openmp_environment({"OMP_NUM_THREADS": " 2 ", "kmp_blocktime": "",
            "VCOMP_UNKNOWN_EXTENSION": "raw", "NOT_RELEVANT": "ignored"}, source="fixture child launch mapping")
        self.assertEqual(result["raw_values"]["OMP_NUM_THREADS"], " 2 ")
        self.assertEqual(result["raw_values"]["kmp_blocktime"], "")
        self.assertIn("VCOMP_UNKNOWN_EXTENSION", result["raw_values"])
        self.assertEqual(result["known_keys"]["OMP_WAIT_POLICY"]["status"], "unset")
        self.assertFalse(result["runtime_defaults_inferred"])

    def test_missing_launch_environment_is_unknown_not_current_environment(self):
        result = w.capture_openmp_environment(None, source="not captured")
        self.assertEqual(result["status"], "unknown")
        self.assertIsNone(result["raw_values"])

    def test_snapshot_delegates_existing_counter_primitive(self):
        calls = []
        def reader(handle, **kwargs):
            calls.append(handle)
            return snapshot(10, 11, 7)["cpu_counter_snapshot"]
        result = w.actor_cpu_snapshot(self.actor, clock=Clock(), windows=True,
                                      reader=reader, identity_reader=lambda handle: 99)
        self.assertEqual(calls, [123])
        self.assertEqual(result["cpu_counter_snapshot"]["process"]["values"]["user_time_100ns"], 7)
        self.assertFalse(result["process_quiescence_inferred"])

    def test_wrong_handle_pid_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "PID differs"):
            w.actor_cpu_snapshot(self.actor, clock=Clock(), windows=True, identity_reader=lambda handle: 100)

    def test_positive_cpu_after_boundary_is_evidence_but_not_quiescence(self):
        result = w.cpu_counter_window(snapshot(10, 11, 5), snapshot(20, 21, 9), boundary_tick=8)
        self.assertEqual(result["activity"]["values"]["cpu_time_100ns"], 4)
        self.assertEqual(result["possible_elapsed_ticks_min"], 9)
        self.assertEqual(result["possible_elapsed_ticks_max"], 11)
        self.assertIn("wholly_after_boundary", result["evidence"])
        self.assertFalse(result["process_quiescence_inferred"])

    def test_zero_cpu_delta_does_not_prove_silence(self):
        result = w.cpu_counter_window(snapshot(10, 11, 5), snapshot(20, 21, 5), boundary_tick=8)
        self.assertIn("silence_not_proven", result["evidence"])

    def test_boundary_straddling_window_cannot_localize_cpu_activity(self):
        result = w.cpu_counter_window(snapshot(10, 11, 5), snapshot(20, 21, 9), boundary_tick=15)
        self.assertFalse(result["entire_outer_envelope_after_boundary"])
        self.assertIn("not_wholly_after", result["evidence"])

    def test_unknown_process_identity_prevents_positive_evidence_claim(self):
        result = w.cpu_counter_window(snapshot(10, 11, 5, identity=False), snapshot(20, 21, 9))
        self.assertEqual(result["evidence"], "unknown_counter_or_identity")

    def test_module_query_failure_is_unknown_and_never_a_loaded_proof(self):
        result = w.snapshot_loaded_modules(self.actor, clock=Clock(), windows=False)
        self.assertEqual(result["status"], "unknown")
        self.assertFalse(result["observed_loaded_path_proof"])

    def test_injected_module_paths_bind_disk_bytes_but_do_not_claim_native_proof(self):
        (ROOT / "build").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="activity-fixture-", dir=ROOT / "build") as directory:
            path = Path(directory) / "VCOMP140.DLL"
            path.write_bytes(b"fixture DLL bytes; not an actual PE")
            calls = []
            def modules():
                calls.append(1)
                return dict(query_begin_tick=1, query_end_tick=2,
                            modules=[dict(base_address=4096, path=str(path))])
            result = w.snapshot_loaded_modules(self.actor, clock=Clock(), windows=True,
                identity_reader=lambda handle: 99, module_reader=modules,
                version_reader=lambda target: dict(file_version=[1, 2, 3, 4]))
            self.assertEqual(len(calls), 2)
            self.assertEqual(result["modules"][0]["on_disk_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(result["modules"][0]["on_disk_version"]["file_version"], [1, 2, 3, 4])
            self.assertTrue(result["module_paths_observed_by_provider"])
            self.assertFalse(result["observed_loaded_path_proof"])
            self.assertFalse(result["loaded_image_memory_bytes_verified"])

    def test_module_hash_budget_exhaustion_remains_unknown(self):
        (ROOT / "build").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="activity-fixture-", dir=ROOT / "build") as directory:
            path = Path(directory) / "runtime.dll"
            path.write_bytes(b"too large")
            result = w.snapshot_loaded_modules(self.actor, clock=Clock(), windows=True,
                identity_reader=lambda handle: 99, module_reader=lambda: dict(modules=[dict(base_address=1, path=str(path))]),
                max_total_hash_bytes=0)
            self.assertEqual(result["status"], "partial")
            self.assertIsNone(result["modules"][0]["on_disk_sha256"])
            self.assertIn("budget", result["modules"][0]["disk_unknown"]["hash"])

    def test_sampler_is_configured_but_never_started_by_factory(self):
        sampler = w.make_counter_sampler([self.actor], clock=Clock())
        self.assertIsInstance(sampler, counters.ProcessCounterSampler)
        self.assertFalse(sampler._started)

    def test_cheap_module_paths_do_not_require_disk_files_or_versions(self):
        calls = []
        def provider():
            calls.append(1)
            return dict(modules=[dict(base_address=1, path="unreadable-or-nonexistent-fixture.dll")])
        result = w.observe_loaded_module_paths(self.actor, clock=Clock(), windows=True,
            identity_reader=lambda handle: 99, module_reader=provider)
        self.assertEqual(result["status"], "observed")
        self.assertEqual(len(calls), 2)
        self.assertFalse(result["on_disk_bytes_hashed"])
        self.assertFalse(result["file_versions_read"])
        self.assertFalse(result["observed_loaded_path_proof"])

    def test_module_inventory_change_is_explicit_not_an_atomic_claim(self):
        calls = []
        def provider():
            calls.append(1)
            return dict(modules=[dict(base_address=1, path="first.dll" if len(calls) == 1 else "second.dll")])
        result = w.observe_loaded_module_paths(self.actor, clock=Clock(), windows=True,
            identity_reader=lambda handle: 99, module_reader=provider)
        self.assertEqual(result["status"], "partial")
        self.assertFalse(result["same_module_inventory_before_after"])
        self.assertFalse(result["atomic_module_snapshot_claimed"])

    def test_file_version_unavailable_is_unknown_without_a_default(self):
        result = w.read_file_version("unused.dll", windows=False)
        self.assertEqual(result["status"], "unknown")
        self.assertIsNone(result["value"])


if __name__ == "__main__":
    unittest.main()
