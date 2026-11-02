"""Collector provenance, continuous protocol and cleanup using fixtures only."""
from copy import deepcopy
import ctypes
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "profiling"))
from resource_calibration import collect
from resource_calibration.schedule import make_schedules


class FakeClock:
    frequency_hz = 1_000_000_000
    def __init__(self):
        self.value = 100
    def now(self):
        self.value += 100
        return self.value
    def metadata(self):
        return {"clock_id": "fixture", "frequency_hz": self.frequency_hz}


class CollectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scratch = ROOT / "build/resource-collector-tests"
        scratch.mkdir(parents=True, exist_ok=True)
        cls.temporary = tempfile.TemporaryDirectory(dir=scratch)
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)

    @staticmethod
    def native_fixture(stamped=True):
        from highlevel_accuracy.bridge import RUNTIME_FLAGS
        action = {"kind": "COMPUTE", "device": "gpu", "target_id": "pv"}
        program = {"program_sha256": "a" * 64, "actions": [action], "bindings": [{"operation_id": "pv", "kind": "attention_pv"}]}
        item = {"iteration": 0, "begin_tick": 2000, "end_tick": 200_002_000, "elapsed_ns": 200_000_000}
        block = {"block_index": 0, "begin_tick": 1000, "end_tick": 200_003_000, "wall_elapsed_ns": 200_002_000,
                 "minimum_elapsed_satisfied": True, "fixed_iteration_count_satisfied": None, "target_diagnostics": [],
                 "iterations": [item], "iteration_count": 1, "completed_work_ns": 200_000_000,
                 "action_intervals": [{**item, "action_index": 0}] if stamped else []}
        native = {"format_version": "state-calibration-native-v1", "program_sha256": program["program_sha256"], "process_id": 123,
                  "mode": "isolated", "target_action_index": 0, "actions": deepcopy(program["actions"]), "target": {"action": deepcopy(action)},
                  "observed_runtime_flags": dict(RUNTIME_FLAGS), "execution_boundary": "legacy_double_sync",
                  "clock": {"clock_id": "fixture", "frequency_hz": 1_000_000_000}, "iteration_policy": "minimum_wall_time",
                  "requested_iterations_per_block": None, "full_fixed_count": None, "graceful_stop_requested": True,
                  "requested_blocks": 1000, "minimum_block_ms": 200,
                  "handshake": {"ready_file_requested": True, "stop_file_requested": True},
                  "instrumentation": "per_iteration_and_per_action_host_completed_work" if stamped else "per_iteration_host_completed_work_only",
                  "ready_tick": 300, "blocks": [block], "stop_observed_tick": 200_004_000}
        return native, program

    def test_continuous_native_protocol_accepts_explicit_stop_and_bare_background(self):
        for stamped in (False, True):
            native, program = self.native_fixture(stamped)
            collect.verify_native_timeline(native, program, process_id=123, target=0, mode="isolated", clock=native["clock"], stamped=stamped)

    def test_final_stopped_block_can_be_short_but_prior_blocks_cannot(self):
        native, program = self.native_fixture()
        block = native["blocks"][0]
        block.update(end_tick=10_000, wall_elapsed_ns=9000, minimum_elapsed_satisfied=False, completed_work_ns=1000)
        for item in (block["iterations"][0], block["action_intervals"][0]):
            item.update(end_tick=3000, elapsed_ns=1000)
        collect.verify_native_timeline(native, program, process_id=123, target=0, mode="isolated", clock=native["clock"])
        second = deepcopy(block)
        second.update(block_index=1, begin_tick=11000, end_tick=20000)
        for item in (second["iterations"][0], second["action_intervals"][0]):
            item.update(begin_tick=12000, end_tick=13000)
        native["blocks"].append(second)
        with self.assertRaisesRegex(ValueError, "only stopped final"):
            collect.verify_native_timeline(native, program, process_id=123, target=0, mode="isolated", clock=native["clock"])

    def test_native_identity_counts_timing_and_boundary_corruption_rejected(self):
        mutations = [lambda x: x.update(graceful_stop_requested=False), lambda x: x.update(process_id=999),
                     lambda x: x.update(iteration_policy="fixed_count"), lambda x: x.update(execution_boundary="async_events"),
                     lambda x: x["blocks"][0].update(completed_work_ns=1), lambda x: x["blocks"][0].update(wall_elapsed_ns=1),
                     lambda x: x["blocks"][0]["iterations"][0].update(elapsed_ns=1),
                     lambda x: x["blocks"][0]["action_intervals"][0].update(action_index=9),
                     lambda x: x.update(stop_observed_tick=1), lambda x: x["handshake"].update(ready_file_requested=False)]
        for index, mutate in enumerate(mutations):
            native, program = self.native_fixture()
            mutate(native)
            with self.subTest(index=index), self.assertRaises(ValueError):
                collect.verify_native_timeline(native, program, process_id=123, target=0, mode="isolated", clock=native["clock"])

    def test_all_observations_retained_and_annotated(self):
        native, _ = self.native_fixture()
        resources = make_schedules()[0]["phases"][0]["resources"]
        events = [{"phase_index": 0, "steady_begin_tick": 2000, "steady_end_tick": 200_002_000, "requested_resources": resources}]
        job = {"id": "train-0", "split": "train", "seed": 27}
        rows = collect.extract_observations(native, events, job=job)
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["metric"] for row in rows}, {"iteration", "target"})
        self.assertTrue(all(row["phase"]["kind"] == "steady" for row in rows))
        events[0]["steady_begin_tick"] = 2001
        rows = collect.extract_observations(native, events, job=job)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["phase"]["kind"] == "transition_or_outside" for row in rows))
        self.assertTrue(all(row["total_telemetry_used_as_predictor"] is False for row in rows))

    def test_inventory_digest_independent_of_mapping_order(self):
        self.assertEqual(collect.inventory_digest({"a": "1", "b": "2"}), collect.inventory_digest({"b": "2", "a": "1"}))
        self.assertNotEqual(collect.inventory_digest({"a": "1"}), collect.inventory_digest({"a": "2"}))

    def test_owned_child_readiness_failure_stops_exact_owned_process(self):
        process = mock.Mock(pid=451)
        process.wait.return_value = 0
        process.poll.return_value = 0
        handles = (mock.Mock(), mock.Mock())
        directory = self.directory / "failed-ready"
        with mock.patch.object(collect.run, "launch", return_value=(process, *handles, {"applied": False})), \
                mock.patch.object(collect, "_json_until_ready", side_effect=TimeoutError("fixture")):
            with self.assertRaises(TimeoutError):
                collect.OwnedChild("owned", directory, ["fixture"], {}, FakeClock(), affinity_mask=3)
        self.assertTrue((directory / "stop.json").is_file())
        self.assertTrue((directory / "process.json").is_file())
        self.assertEqual(process.wait.call_count, 1)
        self.assertTrue(all(handle.close.called for handle in handles))

    def test_owned_child_forced_cleanup_is_explicit_not_success(self):
        process = mock.Mock(pid=452)
        process.wait.side_effect = [subprocess.TimeoutExpired("fixture", 1), 0]
        process.poll.return_value = 0
        handles = (mock.Mock(), mock.Mock())
        ready = {"process_id": 452, "clock": FakeClock().metadata()}
        with mock.patch.object(collect.run, "launch", return_value=(process, *handles, {})), \
                mock.patch.object(collect, "_json_until_ready", return_value=ready):
            child = collect.OwnedChild("owned", self.directory / "forced-cleanup", ["fixture"], {}, FakeClock(), affinity_mask=3)
            result = child.stop()
            self.assertTrue(result["forced_termination"])
            self.assertTrue(result["process_exited_verified"])
            process.terminate.assert_called_once()
            self.assertIs(child.stop(), result)

    def test_host_memory_worker_holds_touches_and_releases_without_deps(self):
        directory = self.directory / "host-memory"
        directory.mkdir()
        ready, stop, result = collect._child_paths(directory)
        def observe_sleep(_):
            self.assertTrue(Path(ready).is_file())
            self.assertFalse(Path(result).exists())
            Path(stop).touch()
        with mock.patch.object(collect, "_host_available_memory", return_value=1 << 30), mock.patch.object(collect.time, "sleep", side_effect=observe_sleep):
            row = collect.memory_worker(device="cpu", size=8193, ready=Path(ready), stop=Path(stop), result=Path(result))
        self.assertEqual(row["status"], "released")
        self.assertEqual(row["allocated_bytes"], 8193)
        self.assertTrue(row["released"])
        self.assertFalse(row["resident_bytes_verified"])
        self.assertLessEqual(row["allocation_end_tick"], row["release_begin_tick"])

    def test_host_memory_guard_fails_loudly_without_allocation_or_readiness(self):
        directory = self.directory / "memory-limit"
        directory.mkdir()
        ready, stop, result = map(Path, collect._child_paths(directory))
        with mock.patch.object(collect, "_host_available_memory", return_value=1024):
            row = collect.memory_worker(device="cpu", size=1024, ready=ready, stop=stop, result=result)
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["allocated_bytes"], 0)
        self.assertFalse(ready.exists())
        self.assertTrue(result.exists())

    def test_cuda_memory_fixture_explicit_alloc_memset_sync_and_free(self):
        directory = self.directory / "cuda-memory"
        directory.mkdir()
        ready, stop, result = map(Path, collect._child_paths(directory))
        runtime = directory / "fixture-runtime.dll"
        runtime.write_bytes(b"not a library; mocked explicit API")
        api = mock.Mock()
        api.cudaSetDevice.return_value = api.cudaMemset.return_value = api.cudaDeviceSynchronize.return_value = api.cudaFree.return_value = 0
        def memory_info(free, total):
            ctypes.cast(free, ctypes.POINTER(ctypes.c_size_t))[0] = 1 << 30
            ctypes.cast(total, ctypes.POINTER(ctypes.c_size_t))[0] = 2 << 30
            return 0
        def malloc(pointer, size):
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.c_void_p(1234)
            return 0
        api.cudaMemGetInfo.side_effect = memory_info
        api.cudaMalloc.side_effect = malloc
        with mock.patch.object(collect, "_cuda_api", return_value=api), mock.patch.object(collect.time, "sleep", side_effect=lambda _: stop.touch()):
            row = collect.memory_worker(device="gpu", size=4096, ready=ready, stop=stop, result=result, cuda_runtime=runtime)
        self.assertEqual(row["status"], "released")
        self.assertEqual(row["allocated_bytes"], 4096)
        self.assertIsNone(row["context_and_allocator_overhead_bytes"])
        self.assertEqual(api.cudaFree.call_count, 2)
        api.cudaMalloc.assert_called_once()
        api.cudaMemset.assert_called_once()

    def test_collect_heldout_rejects_missing_freeze_before_any_launch(self):
        manifest = {"schema": collect.SCHEMA, "sources": {}, "source_inventory_sha256": collect.inventory_digest({})}
        directory = self.directory / "heldout-no-freeze"
        directory.mkdir()
        collect.run.save(directory / "manifest.json", manifest)
        with mock.patch.object(collect, "execute_timeline") as execute, self.assertRaisesRegex(ValueError, "frozen"):
            collect.collect_campaign(directory, "heldout")
        execute.assert_not_called()

    def test_clipped_invocation_union_is_not_a_utilization_estimate(self):
        receipt = collect._clipped_coverage([(0, 12), (11, 20), (30, 100)], 10, 40)
        self.assertEqual(receipt["covered_ticks"], 20)
        self.assertEqual(receipt["window_ticks"], 30)
        self.assertEqual(receipt["overlapping_invocation_count"], 3)
        self.assertIn("not active utilization", receipt["scope"])

    def test_allocation_lifetime_and_replay_evidence_are_separate(self):
        memory = mock.Mock(process=mock.Mock(pid=10), directory=Path("memory"))
        replay = mock.Mock(process=mock.Mock(pid=11), directory=Path("replay"))
        events = [{"phase_index": 0, "steady_begin_tick": 100, "steady_end_tick": 200,
                   "active_resource_pids": {"host_memory": 10, "gpu_replay": 11}}]
        results = [{"allocation_end_tick": 90, "release_begin_tick": 210, "requested_bytes": 64, "allocated_bytes": 64},
                   {"blocks": [{"iterations": [{"begin_tick": 80, "end_tick": 150}, {"begin_tick": 170, "end_tick": 220}]}]}]
        with mock.patch.object(collect.run, "read", side_effect=results):
            collect.attach_resource_evidence(events, [memory, replay])
        evidence = events[0]["resource_evidence"]
        self.assertTrue(evidence["host_memory"]["allocation_lifetime_covers_phase"])
        self.assertFalse(evidence["host_memory"]["resident_bytes_verified"])
        self.assertEqual(evidence["gpu_replay"]["covered_ticks"], 80)
        results[0]["release_begin_tick"] = 190
        with mock.patch.object(collect.run, "read", side_effect=results), self.assertRaisesRegex(ValueError, "lifetime"):
            collect.attach_resource_evidence(events, [memory, replay])

    def test_memory_worker_rejects_alias_paths_and_nonfinite_deadlines(self):
        for deadline in (True, float("nan"), float("inf"), 0, 181):
            with self.subTest(deadline=deadline), self.assertRaises(ValueError):
                collect.memory_worker(device="cpu", size=64, ready=Path("fixture-ready"), stop=Path("fixture-stop"),
                                      result=Path("fixture-result"), timeout_seconds=deadline)
        with self.assertRaises(ValueError):
            collect.memory_worker(device="cpu", size=64, ready=Path("fixture-ready"), stop=Path("fixture-stop"), result=Path("fixture-ready"))

    def test_replay_launch_uses_unfixed_finite_native_loop_and_prelaunch_flags(self):
        native, program = self.native_fixture()
        manifest = {"runtime_dirs": [], "binary": "fixture-probe", "protocol": {"gpu_replay_affinity_mask": 12, "child_stop_timeout_seconds": 30}}
        with mock.patch.object(collect.run, "child_environment", return_value={"STATE_PROBE_FIXED_ITERATIONS": "256", "NVIDIA_TF32_OVERRIDE": "0"}), \
                mock.patch.object(collect, "OwnedChild") as child:
            collect.start_replay_child(Path("fixture-child"), device="gpu", manifest=manifest, binding={"path": "fixture-program"}, program=program, clock=FakeClock())
        name, directory, command, env, _ = child.call_args.args
        self.assertEqual(name, "gpu_replay")
        self.assertEqual(command[2:8], ["isolated", "0", "20", "1000", "200", "bare"])
        self.assertNotIn("STATE_PROBE_FIXED_ITERATIONS", env)
        self.assertEqual(env["NVIDIA_TF32_OVERRIDE"], "0")
        self.assertIn("STATE_PROBE_READY_FILE", env)
        self.assertIn("STATE_PROBE_STOP_FILE", env)

    def test_prepare_guards_reject_before_any_output_is_created(self):
        for kwargs in ({"device_memory_bytes": 64, "cuda_runtime": None}, {"phase_duration_ms": 10000, "device_memory_bytes": 0},
                       {"mode": "parallel", "device_memory_bytes": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                collect.prepare_campaign(self.directory / "never-created", Path("absent-native"), [], **kwargs)
        self.assertFalse((self.directory / "never-created").exists())

    def test_sampler_failure_still_stops_other_sampler_and_writes_failure_receipts(self):
        from resource_calibration import counters as counter_module
        native, program = self.native_fixture()
        sampler = mock.Mock()
        sampler.start.return_value = sampler
        sampler.write_jsonl.side_effect = OSError("fixture raw write failure")
        sampler.metadata.return_value = {"stopped": True}
        process = mock.Mock(pid=987, _handle=123)
        process.poll.return_value = None
        child = mock.Mock(process=process, record={"ready": {}})
        child.stop.side_effect = lambda **_: child.record.update(returncode=0, process_exited_verified=True)
        observed = mock.Mock()
        observed.start.return_value = observed
        observed.metadata.return_value = {"stopped": True}
        clock = FakeClock()
        job = {"id": "sampler-write-failure", "program_key": "p", "mode": "isolated", "schedule": make_schedules(smoke=True)[0]}
        manifest = {"programs": {"p": {"path": "fixture-program", "sha256": "a", "reference": "fixture-reference", "reference_sha256": "b"}},
                    "runtime_dirs": [], "binary": "fixture-native", "source_inventory_sha256": "c",
                    "protocol": {"process_timeout_seconds": 180, "warmup": 20, "telemetry_interval_ms": 50,
                                 "main_affinity_mask": 3, "child_stop_timeout_seconds": 30}}
        def read(path):
            return program if str(path) == "fixture-program" else {}
        with mock.patch.object(collect.run, "sha", side_effect=lambda path: "a" if str(path) == "fixture-program" else "b"), \
                mock.patch.object(collect.run, "read", side_effect=read), mock.patch.object(collect, "NativeClock", return_value=clock), \
                mock.patch.object(collect, "DeviceTelemetrySampler", return_value=sampler), mock.patch.object(counter_module, "ProcessCounterSampler", return_value=observed), \
                mock.patch.object(collect, "OwnedChild", return_value=child), mock.patch.object(collect, "_assert_alive", side_effect=RuntimeError("fixture failure before phases")):
            row = collect.execute_timeline(self.directory, manifest, job)
        self.assertEqual(row["status"], "failed")
        observed.stop.assert_called_once()
        sampler.stop.assert_called_once()
        self.assertTrue(row["cleanup_failure"])
        self.assertIn("fixture raw write failure", str(row["cleanup_errors"]))
        self.assertTrue((self.directory / "jobs" / job["id"] / "execution.json").is_file())
        self.assertTrue((self.directory / "jobs" / job["id"] / "timeline.json").is_file())


if __name__ == "__main__":
    unittest.main()
