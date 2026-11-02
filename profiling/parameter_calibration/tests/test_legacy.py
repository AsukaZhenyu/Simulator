"""Synthetic historical artifacts test the read-only reference importer."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from profiling.parameter_calibration.legacy import import_legacy_resource_campaign
from profiling.parameter_calibration.observations import digest


class HistoricalImportTests(unittest.TestCase):
    def setUp(self):
        base = Path(__file__).resolve().parents[1] / "results"
        base.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directory = self.root / "jobs" / "train-0"
        (self.directory / "main").mkdir(parents=True)
        self.resources = dict(cpu_replay=False, gpu_replay=False,
                              host_memory_bytes=0, device_memory_bytes=0)
        self.job = dict(id="train-0", split="train", seed=7, mode="mixed")
        self.manifest = dict(schema="resource-trajectory-campaign-0.1", sources={},
            source_inventory_sha256=digest({}), jobs=[self.job], protocol_id="synthetic-historical-only")
        phase = dict(kind="steady", phase_index=0, offered_resources=self.resources)
        target = dict(id="train-0-target-0", run_id="train-0", split="train", seed=7,
            metric="target", begin_tick=10, end_tick=20, elapsed_ns=10, phase=phase)
        outside = dict(target, id="outside", begin_tick=30, end_tick=40,
                       phase=dict(kind="transition_or_outside", phase_index=None, offered_resources=None))
        self.raw = [target, outside, dict(target, id="iteration", metric="iteration")]
        self.timeline = dict(clock=dict(clock_id="synthetic_clock", frequency_hz=1_000_000_000),
            events=[dict(steady_begin_tick=0, steady_end_tick=25, phase_index=0,
                         requested_resources=self.resources)])
        self.native = dict(clock=dict(clock_id="synthetic_clock", frequency_hz=1_000_000_000),
                           process_id=42, execution_boundary="legacy_double_sync")
        self.record = dict(job=self.job, status="measured", cleanup_verified=True,
            summary=dict(native_numeric_passed=True), process_id=42,
            started_utc="2026-10-18T00:00:00Z", completed_utc="2026-10-18T00:00:01Z")
        self.write()

    def save(self, path, value):
        path.write_text(json.dumps(value), encoding="utf-8", newline="\n")

    def write(self):
        for name, value in (("observations.json", self.raw), ("timeline.json", self.timeline),
                            ("numerical.json", {"passed": True}), ("main/result.json", self.native)):
            self.save(self.directory / name, value)
        self.record["artifacts_sha256"] = {
            name: hashlib.sha256((self.directory / name).read_bytes()).hexdigest()
            for name in ("observations.json", "timeline.json", "numerical.json", "main/result.json")}
        self.save(self.directory / "execution.json", self.record)
        self.save(self.root / "train-rows.json", [self.record])
        self.save(self.root / "manifest.json", self.manifest)

    def test_total_time_remains_reference_and_transition_retained_as_count(self):
        result = import_legacy_resource_campaign(self.root)
        self.assertEqual(len(result.observations), 1)
        row = result.observations[0]
        self.assertEqual(row.kind, "reference_duration")
        self.assertEqual(row.timing_scope, "whole_action_completed")
        self.assertEqual(row.value, 10)
        self.assertTrue(row.source_verified)
        self.assertEqual(result.receipt["target_sample_count"], 2)
        self.assertEqual(result.receipt["transition_or_outside_samples"], 1)
        self.assertFalse(result.receipt["physical_parameters_estimated"])
        self.assertFalse(result.receipt["legacy_ridge_fit_imported"])
        self.assertFalse(result.receipt["accuracy_accepted"])

    def test_changed_raw_bytes_rejected(self):
        (self.directory / "observations.json").write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "bound artifact changed"):
            import_legacy_resource_campaign(self.root)

    def test_missing_consumed_source_hash_rejected(self):
        self.record["artifacts_sha256"].pop("timeline.json")
        self.save(self.directory / "execution.json", self.record)
        self.save(self.root / "train-rows.json", [self.record])
        with self.assertRaisesRegex(ValueError, "lacks a hash binding"):
            import_legacy_resource_campaign(self.root)

    def test_wrong_elapsed_not_converted_to_rate_or_repaired(self):
        self.raw[0]["elapsed_ns"] = 11
        self.write()
        with self.assertRaisesRegex(ValueError, "elapsed time does not reconstruct"):
            import_legacy_resource_campaign(self.root)

    def test_state_assignment_reconstructed_not_trusted(self):
        self.raw[0]["phase"]["phase_index"] = 9
        self.write()
        with self.assertRaisesRegex(ValueError, "state label disagrees"):
            import_legacy_resource_campaign(self.root)

    def test_equal_frequency_different_clock_identity_rejected(self):
        self.native["clock"]["clock_id"] = "other_clock"
        self.write()
        with self.assertRaisesRegex(ValueError, "clocks disagree"):
            import_legacy_resource_campaign(self.root)

    def test_failed_numerics_not_silently_dropped(self):
        self.record["summary"]["native_numeric_passed"] = False
        self.write()
        with self.assertRaisesRegex(ValueError, "numerical validation failed"):
            import_legacy_resource_campaign(self.root)

    def test_read_only_import_does_not_modify_raw_files(self):
        before = {str(path.relative_to(self.root)): path.read_bytes()
                  for path in self.root.rglob("*") if path.is_file()}
        import_legacy_resource_campaign(self.root)
        after = {str(path.relative_to(self.root)): path.read_bytes()
                 for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
