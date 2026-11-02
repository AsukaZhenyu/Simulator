"""Exact declared record consumption; no hardware, prediction or fitting."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from single_layer_calibration import run


class ExactRecordSetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.out = Path(self.temporary.name)
        self.case = {"id": "case-a"}
        self.manifest = {"manifest_sha256": "a" * 64, "software_sha256": "b" * 64,
            "cases": [self.case, {"id": "case-b"}],
            "protocol": {"polls": ["default", "0"], "independent_processes": 3}}
        rows = []
        for index, run_id in enumerate(sorted(run.expected_runs(self.manifest, "profile"))):
            row = dict(run_id=run_id, pid=100 + index, seed=1000 + index, status="measured")
            record = dict(run_id=run_id, process_id=row["pid"], seed=row["seed"],
                          software_sha256=self.manifest["software_sha256"])
            self.write(self.out / "jobs" / run_id / "record.json", record)
            rows.append(row)
        self.phase = dict(split="profile", records=rows, successful=len(rows), failed=0,
            manifest_sha256=self.manifest["manifest_sha256"], sources_and_runtime_verified_at_phase_end=True)
        self.write(self.out / "profile-phase.json", self.phase)
        self.prefix = "case-a--poll-default--profile-"

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n", encoding="utf-8", newline="\n")

    def selected_ids(self):
        return sorted(row["run_id"] for row in self.phase["records"] if row["run_id"].startswith(self.prefix))

    def consume(self):
        return run.records_for(self.out, self.case, "default", "profile", self.manifest)

    def test_extra_matching_record_rejected_even_when_planned_processes_exist(self):
        self.assertEqual([row["run_id"] for row in self.consume()], self.selected_ids())
        extra_id = "case-a--poll-default--profile-99--context-trace"
        self.write(self.out / "jobs" / extra_id / "record.json", dict(run_id=extra_id))
        with self.assertRaisesRegex(ValueError, "consumed record set"):
            self.consume()

    def test_missing_matching_record_rejected_instead_of_using_survivors(self):
        missing = self.out / "jobs" / self.selected_ids()[0] / "record.json"
        missing.unlink()
        with self.assertRaisesRegex(ValueError, "consumed record set"):
            self.consume()

    def test_record_from_another_case_cannot_replace_a_declared_record(self):
        wanted = self.out / "jobs" / self.selected_ids()[0] / "record.json"
        other = next(row["run_id"] for row in self.phase["records"]
                     if row["run_id"].startswith("case-b--poll-default--profile-"))
        replacement = deepcopy(run.read(self.out / "jobs" / other / "record.json"))
        self.write(wanted, replacement)
        with self.assertRaisesRegex(ValueError, "launched process inventory"):
            self.consume()

    def test_record_from_another_poll_cannot_replace_a_declared_record(self):
        wanted = self.out / "jobs" / self.selected_ids()[0] / "record.json"
        other = next(row["run_id"] for row in self.phase["records"]
                     if row["run_id"].startswith("case-a--poll-0--profile-"))
        replacement = deepcopy(run.read(self.out / "jobs" / other / "record.json"))
        self.write(wanted, replacement)
        with self.assertRaisesRegex(ValueError, "launched process inventory"):
            self.consume()


if __name__ == "__main__":
    unittest.main()
