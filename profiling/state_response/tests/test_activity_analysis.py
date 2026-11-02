"""Synthetic offline diagnostics; deliberately no process or hardware queries."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from profiling.state_response import activity_analysis as a
from profiling.resource_calibration import counters


def fixture():
    clock = dict(clock_id="fixture-QPC", frequency_hz=10_000_000)
    go = dict(clock=clock, start_tick=100, end_tick=400, memory=dict(dram_hold_bytes=0, vram_hold_bytes=0),
        phases=[dict(phase_index=0, begin_tick=100, end_tick=200, cpu_replay=1, gpu_replay=0),
                dict(phase_index=1, begin_tick=200, end_tick=300, cpu_replay=0, gpu_replay=0),
                dict(phase_index=2, begin_tick=300, end_tick=400, cpu_replay=1, gpu_replay=1)])
    events = dict(go=go, known_segments=[dict(begin_tick=100, end_tick=500)])
    pids = {"cpu-worker": 100, "gpu-worker": 200, "main": 300}
    workers = {actor: dict(role=actor, process_id=pid, go_schedule=go,
        blocks=[dict(iterations=[dict(iteration=0, enabled_check_tick=190, begin_tick=190, end_tick=220),
                                 dict(iteration=1, enabled_check_tick=320, begin_tick=320, end_tick=430)])])
               for actor, pid in pids.items()}
    samples, previous = [], {}
    times, cumulative = [50, 70, 170, 190, 210, 230, 250, 270, 290, 310, 450, 470], [0, 0, 1, 2, 3, 4, 5, 5, 6, 7, 8, 9]
    for index, (tick, cpu) in enumerate(zip(times, cumulative)):
        snapshots, activities = {}, {}
        for handle, actor in enumerate(pids, 1):
            raw = dict(schema=counters.SCHEMA, clock=clock, query_begin_tick=tick, query_end_tick=tick + 1,
                process=dict(handle_identity=handle, cpu_counter_begin_tick=tick, cpu_counter_end_tick=tick + 1,
                    values=dict(creation_time_100ns=1, kernel_time_100ns=0, user_time_100ns=cpu)))
            snapshots[actor] = raw
            if actor in previous:
                activities[actor] = counters.delta(previous[actor], raw)
            previous[actor] = raw
        samples.append(dict(sample_index=index, snapshots=snapshots, activities=activities, missed_periods=0, lateness_ticks=0))
    activity = dict(used_as_predictor=False, metadata=dict(clock=clock, interval_ms=10, thread_error=None),
        actors={actor: dict(process_id=pid) for actor, pid in pids.items()}, records=samples)
    dll = "C:\\runtime\\VCOMP140.DLL"
    paths = {actor: dict(status="observed", process_identity=dict(status="verified", observed_pid=pid),
        observed_loaded_path_proof=True, same_module_inventory_before_after=True,
        modules=[dict(path=dll)]) for actor, pid in pids.items()}
    disk = dict(modules=[dict(path=dll, on_disk_sha256="a" * 64,
        file_version=dict(status="observed", value=dict(file_version=[1, 2, 3, 4], product_version=[1, 2, 3, 4])))])
    context = dict(openmp_launch_environment=dict(status="recorded", raw_values={"OMP_NUM_THREADS": "2"}))
    return activity, events, workers, paths, disk, context


class ActivityAnalysisTests(unittest.TestCase):
    def test_delta_reconstruction_retains_true_query_span(self):
        activity, events, workers, _, _, _ = fixture()
        rows, report = a.cpu_windows(activity, actor_id="cpu-worker", expected_pid=100, clock=events["go"]["clock"])
        self.assertEqual(len(rows), len(activity["records"]) - 1)
        self.assertEqual(report["baseline_only_snapshots"], 1)
        self.assertEqual(rows[0]["outer_begin_tick"], 50)
        self.assertEqual(rows[0]["outer_end_tick"], 71)
        self.assertFalse(report["CPU_accounting_resolution_claimed"])

    def test_tampered_cached_activity_is_rejected(self):
        activity, events, _, _, _, _ = fixture()
        activity["records"][1]["activities"]["cpu-worker"]["values"]["cpu_time_100ns"] = 999
        with self.assertRaisesRegex(ValueError, "does not reconstruct"):
            a.cpu_windows(activity, actor_id="cpu-worker", expected_pid=100, clock=events["go"]["clock"])

    def test_state_grouping_excludes_query_windows_crossing_boundary(self):
        activity, events, workers, paths, disk, context = fixture()
        result = a.analyze_run(activity, events, workers, paths, disk, context=context, run_id="profile-fixture")
        phases = result["actors"]["cpu-worker"]["phases"]
        self.assertEqual(phases[0]["activity"]["window_count"], 1)  # 170..191 only.
        self.assertEqual(phases[1]["activity"]["window_count"], 4)  # 210..231 to 270..291.
        self.assertEqual(result["actors"]["cpu-worker"]["pre_GO_idle"]["window_count"], 1)

    def test_tail_requires_both_off_and_last_call_completed(self):
        activity, events, workers, paths, disk, context = fixture()
        result = a.analyze_run(activity, events, workers, paths, disk, context=context, run_id="profile-fixture")
        tail = result["CPU_off_after_last_call"][0]
        self.assertEqual(tail["off_request_tick"], 200)
        self.assertEqual(tail["after_request_and_call_begin_tick"], 220)
        self.assertEqual(tail["actors"]["cpu-worker"]["window_count"], 3)
        self.assertEqual(tail["actors"]["cpu-worker"]["positive_increment_windows"], 2)
        self.assertEqual(tail["actors"]["cpu-worker"]["zero_increment_windows"], 1)
        self.assertFalse(tail["process_quiescence_inferred"])

    def test_missing_old_call_proof_remains_unresolved(self):
        activity, events, workers, paths, disk, context = fixture()
        workers["cpu-worker"]["blocks"][0]["iterations"] = []
        result = a.analyze_run(activity, events, workers, paths, disk, context=context, run_id="profile-fixture")
        self.assertIsNone(result["CPU_off_after_last_call"][0]["after_request_and_call_begin_tick"])
        self.assertEqual(result["CPU_off_after_last_call"][0]["actors"]["cpu-worker"]["window_count"], 0)

    def test_runtime_path_receipt_does_not_claim_mapped_bytes_or_defaults(self):
        _, _, workers, paths, disk, context = fixture()
        result = a.runtime_observations(paths, disk, {actor: worker["process_id"] for actor, worker in workers.items()}, context["openmp_launch_environment"])
        self.assertTrue(result["actors"]["main"]["VCOMP_modules"][0]["native_loaded_path_observed"])
        self.assertFalse(result["actors"]["main"]["VCOMP_modules"][0]["mapped_image_bytes_verified"])
        self.assertFalse(result["runtime_default_values_inferred"])
        self.assertFalse(result["remote_child_environment_read"])

    def test_source_audit_rejects_changed_raw_bytes(self):
        (ROOT / "build").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="activity-analysis-fixture-", dir=ROOT / "build") as directory:
            path = Path(directory) / "raw.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "raw bytes changed"):
                a.SourceAudit().json(path, "0" * 64)

    def runtime_run(self):
        _, _, workers, paths, disk, context = fixture()
        return dict(run_id="profile-fixture", runtime=a.runtime_observations(paths, disk,
            {actor: worker["process_id"] for actor, worker in workers.items()}, context["openmp_launch_environment"]))

    def test_complete_native_path_and_disk_receipts_establish_known_consistency(self):
        result = a.vcomp_consistency([self.runtime_run()])
        self.assertTrue(result["known_receipts_identical"])
        self.assertEqual(result["known_complete_entries"], 3)
        self.assertEqual(result["unknown_entries"], 0)

    def test_all_none_disk_hash_version_does_not_establish_known_consistency(self):
        run = self.runtime_run()
        for actor in run["runtime"]["actors"].values():
            actor["VCOMP_modules"][0]["disk_receipt"] = dict(path=None, on_disk_sha256=None, file_version=None)
            actor["VCOMP_modules"][0]["path"] = None
        result = a.vcomp_consistency([run])
        self.assertFalse(result["known_receipts_identical"])
        self.assertEqual(result["known_complete_entries"], 0)
        self.assertEqual(result["unknown_entries"], 3)

    def test_injected_or_unverified_native_path_cannot_establish_known_consistency(self):
        run = self.runtime_run()
        run["runtime"]["actors"]["main"]["VCOMP_modules"][0]["native_loaded_path_observed"] = False
        result = a.vcomp_consistency([run])
        self.assertTrue(result["known_subset_receipts_identical"])
        self.assertFalse(result["known_receipts_identical"])
        self.assertEqual(result["unknown_entries"], 1)

    def test_unknown_numeric_version_prevents_known_consistency(self):
        run = self.runtime_run()
        run["runtime"]["actors"]["main"]["VCOMP_modules"][0]["disk_receipt"] = deepcopy(run["runtime"]["actors"]["main"]["VCOMP_modules"][0]["disk_receipt"])
        run["runtime"]["actors"]["main"]["VCOMP_modules"][0]["disk_receipt"]["file_version"] = dict(status="unknown", value=None)
        self.assertFalse(a.vcomp_consistency([run])["known_receipts_identical"])

    def test_differing_known_hashes_are_not_consistent(self):
        run = self.runtime_run()
        run["runtime"]["actors"]["main"]["VCOMP_modules"][0]["disk_receipt"] = deepcopy(run["runtime"]["actors"]["main"]["VCOMP_modules"][0]["disk_receipt"])
        run["runtime"]["actors"]["main"]["VCOMP_modules"][0]["disk_receipt"]["on_disk_sha256"] = "b" * 64
        result = a.vcomp_consistency([run])
        self.assertFalse(result["known_receipts_identical"])
        self.assertEqual(result["unknown_entries"], 0)
        self.assertEqual(len(result["distinct_path_hash_version_receipts"]), 2)

    def test_missing_actor_module_is_unknown_instead_of_equal_to_other_missing(self):
        run = self.runtime_run()
        run["runtime"]["actors"] = {}
        result = a.vcomp_consistency([run])
        self.assertFalse(result["known_receipts_identical"])
        self.assertEqual(result["unknown_entries"], 3)


if __name__ == "__main__":
    unittest.main()
