"""Recorded-source evidence remains verifiable after intentional refactoring."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from profiling.parameter_calibration.source_archive import verify_archive


class SourceArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.archive = self.root / "archive"
        (self.archive / "sources").mkdir(parents=True)
        old = b"# original recorded source\n"
        (self.root / "a.py").write_bytes(old)
        (self.archive / "sources/a.py").write_bytes(old)
        sha = hashlib.sha256(old).hexdigest()
        manifest = {"sources": {str((self.root / "a.py").resolve()): sha}}
        self.write(self.root / "manifest.json", manifest)
        self.record = {"schema": "performance-boundary-source-archive-0.1", "git_commit": "fixture",
                       "sources": {"a.py": sha}, "raw_measurements_rewritten": False,
                       "campaigns": {"manifest.json": {"manifest_sha256": hashlib.sha256((self.root / "manifest.json").read_bytes()).hexdigest(),
                                                       "sources": {"a.py": sha}}}}
        self.write(self.archive / "archive.json", self.record)

    def write(self, path, obj):
        path.write_text(json.dumps(obj), encoding="utf-8", newline="\n")

    def test_current_change_is_reported_without_relabeling_old_source(self):
        (self.root / "a.py").write_bytes(b"# refactored\n")
        result = verify_archive(self.archive, self.root)
        self.assertTrue(result["verified"])
        self.assertEqual(result["current_source_changes"], ["a.py"])
        self.assertFalse(result["current_source_equivalence_claimed"])
        self.assertFalse(result["archived_runtime_executed"])

    def test_unchanged_source_is_verified(self):
        result = verify_archive(self.archive, self.root)
        self.assertTrue(result["current_source_equivalence_claimed"])

    def test_archived_source_tamper_is_rejected(self):
        (self.archive / "sources/a.py").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "archived source"):
            verify_archive(self.archive, self.root)

    def test_old_manifest_tamper_is_rejected(self):
        self.write(self.root / "manifest.json", {})
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            verify_archive(self.archive, self.root)

    def test_source_path_escape_is_rejected(self):
        self.record["sources"] = {"../outside.py": "a" * 64}
        self.write(self.archive / "archive.json", self.record)
        with self.assertRaisesRegex(ValueError, "relative"):
            verify_archive(self.archive, self.root)

    def test_raw_rewrite_claim_is_rejected(self):
        self.record["raw_measurements_rewritten"] = True
        self.write(self.archive / "archive.json", self.record)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            verify_archive(self.archive, self.root)


if __name__ == "__main__":
    unittest.main()
