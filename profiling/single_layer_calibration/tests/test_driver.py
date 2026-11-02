"""Non-hardware guards against incomplete or changed calibration evidence."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from single_layer_calibration import run


class DriverTests(unittest.TestCase):
    def setUp(self):
        parent = Path(__file__).resolve().parents[3] / "build/single-layer-driver-tests"
        parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.manifest = {"schema": run.SCHEMA, "manifest_sha256": "c" * 64,
                         "software_sha256": "a" * 64, "cases": [{"id": "case-a"}],
                         "protocol": {"polls": ["default", "0"], "independent_processes": 3}}

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8", newline="\n")

    def phase(self, split="profile"):
        rows = []
        for index, run_id in enumerate(sorted(run.expected_runs(self.manifest, split))):
            row = {"run_id": run_id, "pid": 100 + index, "seed": 1000 + index,
                   "status": "measured", "started_at": "2026-10-18T02:00:00+00:00"}
            record = {"run_id": run_id, "process_id": row["pid"], "seed": row["seed"],
                      "software_sha256": self.manifest["software_sha256"]}
            self.write(self.out / "jobs" / run_id / "record.json", record)
            rows.append(row)
        phase = {"split": split, "records": rows, "successful": len(rows), "failed": 0,
                 "manifest_sha256": self.manifest["manifest_sha256"],
                 "sources_and_runtime_verified_at_phase_end": True}
        self.write(self.out / (split + "-phase.json"), phase)
        return phase

    def barrier(self):
        models = []
        for poll in self.manifest["protocol"]["polls"]:
            path = self.out / "models" / ("case-a--poll-" + poll + ".json")
            self.write(path, {"synthetic_model_fixture": poll})
            models.append({"case_id": "case-a", "poll": poll, "path": str(path), "sha256": run.sha(path)})
        barrier = {"schema": "single-layer-all-models-frozen-1",
                   "manifest_sha256": self.manifest["manifest_sha256"],
                   "software_sha256": self.manifest["software_sha256"],
                   "heldout_used_for_estimation": False,
                   "frozen_at_utc": "2026-10-18T01:00:00+00:00", "models": models}
        self.rewrite_barrier(barrier)
        return barrier

    def rewrite_barrier(self, barrier):
        barrier.pop("barrier_sha256", None)
        barrier["barrier_sha256"] = run.content_sha(barrier)
        self.write(self.out / "all-models-frozen.json", barrier)

    def test_complete_phase_and_barrier_are_readable(self):
        phase, barrier = self.phase(), self.barrier()
        self.assertEqual(run.verified_phase(self.out, self.manifest, "profile"), phase)
        self.assertEqual(run.verified_barrier(self.out, self.manifest), barrier)

    def test_phase_cannot_drop_failed_process_to_leave_survivors(self):
        phase = self.phase()
        phase["records"].pop()
        phase["successful"] -= 1
        self.write(self.out / "profile-phase.json", phase)
        with self.assertRaisesRegex(ValueError, "omits or duplicates"):
            run.verified_phase(self.out, self.manifest, "profile")

    def test_phase_cannot_duplicate_a_successful_run(self):
        phase = self.phase()
        phase["records"][-1] = deepcopy(phase["records"][0])
        self.write(self.out / "profile-phase.json", phase)
        with self.assertRaisesRegex(ValueError, "omits or duplicates"):
            run.verified_phase(self.out, self.manifest, "profile")

    def test_failed_status_cannot_be_hidden_by_counts(self):
        phase = self.phase()
        phase["records"][0]["status"] = "failed"
        self.write(self.out / "profile-phase.json", phase)
        with self.assertRaisesRegex(ValueError, "survivor subset"):
            run.verified_phase(self.out, self.manifest, "profile")

    def test_popen_pid_and_record_pid_must_agree(self):
        phase = self.phase()
        path = self.out / "jobs" / phase["records"][0]["run_id"] / "record.json"
        record = run.read(path)
        record["process_id"] += 1
        self.write(path, record)
        with self.assertRaisesRegex(ValueError, "launched process inventory"):
            run.verified_phase(self.out, self.manifest, "profile")

    def test_final_source_verification_is_required(self):
        phase = self.phase()
        phase["sources_and_runtime_verified_at_phase_end"] = False
        self.write(self.out / "profile-phase.json", phase)
        with self.assertRaisesRegex(ValueError, "final source/runtime"):
            run.verified_phase(self.out, self.manifest, "profile")

    def test_even_resigned_barrier_cannot_drop_or_duplicate_model(self):
        for duplicate in (False, True):
            with self.subTest(duplicate=duplicate):
                barrier = self.barrier()
                barrier["models"][-1:] = [deepcopy(barrier["models"][0])] if duplicate else []
                self.rewrite_barrier(barrier)
                with self.assertRaisesRegex(ValueError, "omits or duplicates"):
                    run.verified_barrier(self.out, self.manifest)

    def test_model_path_cannot_escape_expected_campaign_file(self):
        barrier = self.barrier()
        other = self.out / "unexpected.json"
        other.write_bytes(Path(barrier["models"][0]["path"]).read_bytes())
        barrier["models"][0]["path"] = str(other)
        self.rewrite_barrier(barrier)
        with self.assertRaisesRegex(ValueError, "path or bytes differ"):
            run.verified_barrier(self.out, self.manifest)

    def test_frozen_model_bytes_or_barrier_digest_change_rejected(self):
        barrier = self.barrier()
        Path(barrier["models"][0]["path"]).write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "path or bytes differ"):
            run.verified_barrier(self.out, self.manifest)
        barrier = self.barrier()
        barrier["frozen_at_utc"] = "2026-10-18T00:00:00+00:00"
        self.write(self.out / "all-models-frozen.json", barrier)
        with self.assertRaisesRegex(ValueError, "barrier differs"):
            run.verified_barrier(self.out, self.manifest)

    def test_heldout_cannot_precede_global_barrier(self):
        self.phase("profile")
        phase = self.phase("heldout")
        phase["records"][0]["started_at"] = "2026-10-18T00:59:59+00:00"
        self.write(self.out / "heldout-phase.json", phase)
        self.barrier()
        with patch.object(run, "load_manifest", return_value=self.manifest), \
                patch.object(run, "verify_current"):
            with self.assertRaisesRegex(ValueError, "before the global model barrier"):
                run.evaluate(SimpleNamespace(out=self.out))

    def test_added_runtime_dll_is_detected(self):
        runtime = self.out / "runtime"
        runtime.mkdir()
        library = runtime / "original.dll"
        executable = self.out / "probe.exe"
        library.write_bytes(b"synthetic shared library")
        executable.write_bytes(b"synthetic executable")
        inventory = {"source_sha256": {}, "build_bound_sha256": {},
                     "runtime_sha256": {str(library): run.sha(library), str(executable): run.sha(executable)},
                     "shared_library_inventory": [str(library.resolve())]}
        manifest = {"software": inventory, "python_source_inventory": [],
                    "runtime_dirs": [str(runtime)], "executable": str(executable),
                    "source_runtime_file_stats": run.file_stats(inventory)}
        with patch.object(run, "source_paths", return_value=[]):
            run.verify_current(manifest)
            (runtime / "new-shadowing.dll").write_bytes(b"new library")
            with self.assertRaisesRegex(ValueError, "shared-library inventory changed"):
                run.verify_current(manifest)

    def test_expected_inventory_keeps_all_modes_and_poll_groups(self):
        self.assertEqual(len(run.expected_runs(self.manifest, "profile")), 18)
        self.assertEqual(len(run.expected_runs(self.manifest, "heldout")), 6)
        self.assertTrue(all(name.endswith("--bare-replay") for name in run.expected_runs(self.manifest, "heldout")))

    def test_new_campaign_numerical_seeds_are_disjoint_from_development(self):
        old = {run.numerical_seed(1230, split, repeat, 0)
               for split in ("profile", "heldout") for repeat in range(3)}
        new_profile = {run.numerical_seed(1230, "profile", repeat, 200_000_000) for repeat in range(3)}
        new_heldout = {run.numerical_seed(1230, "heldout", repeat, 200_000_000) for repeat in range(3)}
        self.assertFalse(old & (new_profile | new_heldout))
        self.assertFalse(new_profile & new_heldout)
        self.assertEqual(len(new_profile | new_heldout), 6)

    def test_seed_overflow_cannot_wrap_into_another_cohort(self):
        with self.assertRaisesRegex(ValueError, "uint32"):
            run.numerical_seed(10, "heldout", 0, 2**32)

    def test_invalid_seed_components_fail_instead_of_guessing(self):
        for values in ((0, "test", 0, 0), (0, "profile", 0, -1), (True, "profile", 0, 0)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                run.numerical_seed(*values)


if __name__ == "__main__":
    unittest.main()
