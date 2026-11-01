"""Historical source preservation is separate from rebuilding an old model."""
from copy import deepcopy
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from profiling.service_response_snapshot import audit_snapshot, main


def sha(value):
    return hashlib.sha256(value).hexdigest()


class SourceSnapshotTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='source-preservation-')
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name)
        self.original = self.path / 'current.py'
        self.original.write_bytes(b'original code\n')
        self.binary = self.path / 'runtime.bin'
        self.binary.write_bytes(b'unchanged native runtime')
        self.archive_root = self.path / 'recorded-sources'
        self.archive_root.mkdir()
        self.retained = self.archive_root / 'current.py'
        self.retained.write_bytes(self.original.read_bytes())
        sources = {str(self.original): sha(self.original.read_bytes()), str(self.binary): sha(self.binary.read_bytes())}
        self.manifest = dict(schema='state-response-campaign-1', sources=sources,
            source_inventory_sha256=sha(json.dumps(sources, sort_keys=True, separators=(',', ':')).encode()))
        self.manifest_path = self.path / 'manifest.json'
        self.save(self.manifest_path, self.manifest)
        self.manifest_sha = sha(self.manifest_path.read_bytes())
        self.index = dict(schema='service-response-source-snapshot-1', created_at_utc='2026-10-22T00:00:00+00:00',
            manifest_sha256=self.manifest_sha,
            entries=[dict(original_path=str(self.original), archive_path=str(self.retained), sha256=sources[str(self.original)])])
        self.index_path = self.archive_root / 'archive.json'
        self.save(self.index_path, self.index)

    def save(self, path, value):
        with path.open('w', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, indent=2)

    def audit(self):
        return audit_snapshot(self.manifest_path, expected_manifest_sha256=self.manifest_sha)

    def test_current_source_can_change_without_changing_preserved_history(self):
        self.original.write_bytes(b'new code\n')
        result = self.audit()
        self.assertTrue(result['summary']['source_preservation_verified'])
        self.assertFalse(result['current_code_matches_recorded_version'])
        self.assertEqual(result['summary']['current_original_status_counts'], {'changed':1, 'unchanged':1})
        self.assertFalse(result['parameter_rebuild_verified'])
        self.assertFalse(result['archived_evaluator_executed'])
        self.assertEqual(result['sources'][0]['lookup'], 'archive')

    def test_missing_original_is_allowed_only_when_archived(self):
        self.original.unlink()
        self.assertTrue(self.audit()['summary']['source_preservation_verified'])
        self.binary.unlink()
        result = self.audit()
        self.assertFalse(result['summary']['source_preservation_verified'])
        self.assertEqual(result['summary']['failed_source_count'], 1)

    def test_changed_archive_does_not_fall_back_to_current_original(self):
        self.retained.write_bytes(b'changed archive')
        result = self.audit()
        self.assertFalse(result['summary']['source_preservation_verified'])
        self.assertEqual(result['sources'][0]['consumed']['status'], 'changed')

    def test_missing_archive_does_not_fall_back_to_current_original(self):
        self.retained.unlink()
        result = self.audit()
        self.assertFalse(result['summary']['source_preservation_verified'])
        self.assertEqual(result['sources'][0]['consumed']['status'], 'missing')

    def test_non_archived_runtime_change_fails(self):
        self.binary.write_bytes(b'new native runtime')
        self.assertFalse(self.audit()['summary']['source_preservation_verified'])

    def test_pinned_manifest_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, 'manifest bytes differ'):
            audit_snapshot(self.manifest_path, expected_manifest_sha256='0' * 64)

    def test_internal_inventory_digest_is_verified(self):
        self.manifest['source_inventory_sha256'] = '0' * 64
        self.save(self.manifest_path, self.manifest)
        self.manifest_sha = sha(self.manifest_path.read_bytes())
        with self.assertRaisesRegex(ValueError, 'inventory digest differs'):
            self.audit()

    def test_archive_for_another_manifest_rejected(self):
        self.index['manifest_sha256'] = '0' * 64
        self.save(self.index_path, self.index)
        with self.assertRaisesRegex(ValueError, 'another manifest'):
            self.audit()

    def test_archive_cannot_escape_snapshot_directory(self):
        self.index['entries'][0]['archive_path'] = str(self.original)
        self.save(self.index_path, self.index)
        with self.assertRaisesRegex(ValueError, 'escapes'):
            self.audit()

    def test_duplicate_binding_rejected(self):
        self.index['entries'].append(deepcopy(self.index['entries'][0]))
        self.save(self.index_path, self.index)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.audit()

    def test_changed_recorded_hash_rejected(self):
        self.index['entries'][0]['sha256'] = '0' * 64
        self.save(self.index_path, self.index)
        with self.assertRaisesRegex(ValueError, 'historical inventory'):
            self.audit()

    def test_cli_output_is_write_once(self):
        output = self.path / 'audit.json'
        args = ['--manifest', str(self.manifest_path), '--expected-manifest-sha256', self.manifest_sha,
                '--output', str(output)]
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(args), 0)
        self.assertNotIn(b'\r\n', output.read_bytes())
        with self.assertRaises(FileExistsError):
            main(args)


if __name__ == '__main__':
    unittest.main()
