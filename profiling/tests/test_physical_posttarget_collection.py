"""Small lifecycle tests using owned-process fakes; no hardware is launched."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from profiling import physical_posttarget_collection as collector


class Clock:
    def __init__(self):
        self.tick = 100

    def now(self):
        self.tick += 1
        return self.tick

    def metadata(self):
        return dict(clock_id="windows_qpc", frequency_hz=10_000_000)


class FakeChild:
    def __init__(self, name, *, stop_code=0):
        self.name, self.code, self.exited = name, stop_code, False
        self.process = self
        self.record = dict(name=name, pid=1, forced_termination=False)

    def poll(self):
        return self.code if self.exited else None

    def stop(self, *, cleanup=False):
        self.exited = True
        self.record.update(returncode=self.code, process_exited_verified=True,
                           stop_requested_tick=102, exit_observed_tick=103)
        return self.record


class PostTargetCollectionTests(unittest.TestCase):
    def exercise(self, stop_code):
        with tempfile.TemporaryDirectory(prefix="posttarget-test-", dir=Path(__file__).resolve().parent) as folder:
            root = Path(folder)
            directory = root / "jobs" / "example"
            directory.mkdir(parents=True)
            index = root / "posttarget-job-index.json"
            index.write_text("{}", encoding="utf-8")
            clock = Clock()
            complete = dict(process_id=2, completed_samples=32, sampling_end_ns=1000)
            old_save = collector.campaign.run.save
            wait = lambda child, clock, timeout: complete
            with patch.object(collector.campaign, "OwnedChild", FakeChild), patch.object(
                    collector.campaign, "wait_complete", wait):
                with collector._posttarget_hooks(root, dict(id="example"), index):
                    background = collector.campaign.OwnedChild("cpu-worker", stop_code=stop_code)
                    main = collector.campaign.OwnedChild("main")
                    if stop_code:
                        with self.assertRaisesRegex(ValueError, "normal STOP"):
                            collector.campaign.wait_complete(main, clock, 1)
                        with self.assertRaisesRegex(ValueError, "verified post-target"):
                            collector.campaign.OwnedChild("after-gpu")
                    else:
                        self.assertEqual(collector.campaign.wait_complete(main, clock, 1), complete)
                        self.assertTrue(background.exited)
                        collector.campaign.OwnedChild("after-gpu")
                        row = dict(schema="physical-service-execution-1")
                        collector.campaign.run.save(directory / "execution.json", row)
                        self.assertTrue(row["posttarget_cleanup_applied"])
                        self.assertEqual(row["posttarget_after_reference_scope"], "unloaded_drift_only")
                        receipt = json.loads((directory / "posttarget-cleanup.json").read_text())
                        self.assertEqual(receipt["background_count"], 1)
                        self.assertTrue(receipt["all_backgrounds_exited_before_after_construction"])
                        self.assertFalse(receipt["target_sampling_altered"])
                self.assertIs(collector.campaign.OwnedChild, FakeChild)
                self.assertIs(collector.campaign.wait_complete, wait)
                self.assertIs(collector.campaign.run.save, old_save)

    def test_stop_precedes_after_construction_and_receipt_is_bound(self):
        self.exercise(0)

    def test_failed_stop_blocks_after_construction_and_restores_hooks(self):
        self.exercise(1)


if __name__ == "__main__":
    unittest.main()
